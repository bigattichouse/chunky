#!/usr/bin/env python3
"""
Chunked Inference with Full Checkpoint/Resume System
Enables safe long-form generation with progress saving and resumption
"""

import os
import sys
import json
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Generator, Any
import pickle
import gc

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
import torch.nn.functional as F


@dataclass
class GenerationCheckpoint:
    """Complete checkpoint data for resuming generation"""
    
    # Generation state
    prompt: str
    generated_tokens: List[str]
    current_token_count: int
    target_token_count: int
    
    # Model state (will be saved separately due to size)
    hidden_states_file: Optional[str] = None
    current_sequence_ids: List[int] = None
    
    # Configuration
    model_path: str = ""
    temperature: float = 0.7
    max_vram_gb: float = 4.0
    chunk_layers: int = 1
    
    # Metadata
    timestamp: float = 0.0
    model_config_hash: str = ""
    torch_rng_state: Optional[Dict] = None
    python_rng_state: Optional[Tuple] = None
    
    # Progress tracking
    estimated_time_remaining: float = 0.0
    tokens_per_second: float = 0.0
    checkpoint_version: str = "1.0"
    
    # Performance stats
    total_chunks_processed: int = 0
    average_chunk_time: float = 0.0


class CheckpointManager:
    """Manages saving, loading, and validation of generation checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load checkpoint metadata index"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"checkpoints": [], "last_cleanup": time.time()}
    
    def _save_metadata(self):
        """Save checkpoint metadata index"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_checkpoint_name(self, base_name: str, token_count: int) -> str:
        """Generate unique checkpoint filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_t{token_count:04d}_{timestamp}.ckpt"
    
    def save_checkpoint(self, checkpoint: GenerationCheckpoint, base_name: str, 
                       hidden_states: torch.Tensor, async_save: bool = True) -> str:
        """Save checkpoint with optional async operation"""
        
        # Generate filename
        checkpoint_name = self._generate_checkpoint_name(base_name, checkpoint.current_token_count)
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save hidden states separately (large tensor)
        hidden_states_name = checkpoint_name.replace('.ckpt', '_hidden.pt')
        hidden_states_path = self.checkpoint_dir / hidden_states_name
        checkpoint.hidden_states_file = hidden_states_name
        
        # Update metadata
        checkpoint.timestamp = time.time()
        
        if async_save:
            # Save in background thread
            def _async_save():
                self._do_save_checkpoint(checkpoint, checkpoint_path, hidden_states, hidden_states_path)
            
            thread = threading.Thread(target=_async_save)
            thread.daemon = True
            thread.start()
        else:
            self._do_save_checkpoint(checkpoint, checkpoint_path, hidden_states, hidden_states_path)
        
        return str(checkpoint_path)
    
    def _do_save_checkpoint(self, checkpoint: GenerationCheckpoint, checkpoint_path: Path, 
                           hidden_states: torch.Tensor, hidden_states_path: Path):
        """Actually perform the checkpoint save"""
        try:
            # Save hidden states (compressed)
            torch.save({
                'hidden_states': hidden_states.cpu(),  # Move to CPU for saving
                'dtype': str(hidden_states.dtype),
                'device': str(hidden_states.device)
            }, hidden_states_path, _use_new_zipfile_serialization=True)
            
            # Save checkpoint metadata
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata index
            self.metadata["checkpoints"].append({
                "name": checkpoint_path.name,
                "timestamp": checkpoint.timestamp,
                "tokens": checkpoint.current_token_count,
                "target_tokens": checkpoint.target_token_count,
                "model_path": checkpoint.model_path,
                "prompt_preview": checkpoint.prompt[:100] + "..." if len(checkpoint.prompt) > 100 else checkpoint.prompt
            })
            self._save_metadata()
            
            print(f"\n💾 Checkpoint saved: {checkpoint_path.name} ({checkpoint.current_token_count}/{checkpoint.target_token_count} tokens)")
            
        except Exception as e:
            print(f"\n❌ Error saving checkpoint: {e}")
            # Cleanup partial files
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if hidden_states_path.exists():
                hidden_states_path.unlink()
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[GenerationCheckpoint, torch.Tensor]:
        """Load checkpoint and return checkpoint data + hidden states"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            # Try looking in checkpoint directory
            checkpoint_path = self.checkpoint_dir / checkpoint_path.name
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path.name}")
        
        try:
            # Load checkpoint metadata
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Load hidden states
            hidden_states_path = checkpoint_path.parent / checkpoint.hidden_states_file
            if not hidden_states_path.exists():
                raise FileNotFoundError(f"Hidden states file not found: {hidden_states_path}")
            
            hidden_data = torch.load(hidden_states_path, map_location='cpu')
            hidden_states = hidden_data['hidden_states']
            
            print(f"✅ Checkpoint loaded: {checkpoint.current_token_count}/{checkpoint.target_token_count} tokens")
            print(f"   Prompt: {checkpoint.prompt[:100]}{'...' if len(checkpoint.prompt) > 100 else ''}")
            print(f"   Generated so far: {len(checkpoint.generated_tokens)} tokens")
            
            return checkpoint, hidden_states
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise
    
    def validate_checkpoint(self, checkpoint: GenerationCheckpoint, model_config_hash: str) -> bool:
        """Validate checkpoint compatibility"""
        if checkpoint.checkpoint_version != "1.0":
            print(f"⚠️  Checkpoint version mismatch: {checkpoint.checkpoint_version} vs 1.0")
            return False
        
        if checkpoint.model_config_hash != model_config_hash:
            print(f"⚠️  Model configuration mismatch")
            response = input("Continue anyway? (y/n): ").lower().strip()
            return response == 'y'
        
        return True
    
    def list_checkpoints(self):
        """List available checkpoints"""
        print("\n📋 Available Checkpoints:")
        print("-" * 80)
        
        if not self.metadata["checkpoints"]:
            print("No checkpoints found.")
            return
        
        # Sort by timestamp (newest first)
        checkpoints = sorted(self.metadata["checkpoints"], key=lambda x: x["timestamp"], reverse=True)
        
        for i, ckpt in enumerate(checkpoints):
            timestamp = datetime.fromtimestamp(ckpt["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            progress = f"{ckpt['tokens']}/{ckpt['target_tokens']}"
            print(f"{i+1:2d}. {ckpt['name']}")
            print(f"    Progress: {progress:>10} tokens | {timestamp}")
            print(f"    Prompt: {ckpt['prompt_preview']}")
            print()
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoint files, keeping the most recent N"""
        print(f"🧹 Cleaning up old checkpoints (keeping last {keep_last_n})...")
        
        checkpoints = sorted(self.metadata["checkpoints"], key=lambda x: x["timestamp"], reverse=True)
        
        if len(checkpoints) <= keep_last_n:
            print("No old checkpoints to remove.")
            return
        
        removed_count = 0
        for ckpt in checkpoints[keep_last_n:]:
            try:
                # Remove main checkpoint file
                ckpt_path = self.checkpoint_dir / ckpt["name"]
                if ckpt_path.exists():
                    ckpt_path.unlink()
                
                # Remove hidden states file
                hidden_path = self.checkpoint_dir / ckpt["name"].replace('.ckpt', '_hidden.pt')
                if hidden_path.exists():
                    hidden_path.unlink()
                
                removed_count += 1
                
            except Exception as e:
                print(f"Warning: Could not remove {ckpt['name']}: {e}")
        
        # Update metadata
        self.metadata["checkpoints"] = checkpoints[:keep_last_n]
        self.metadata["last_cleanup"] = time.time()
        self._save_metadata()
        
        print(f"✅ Removed {removed_count} old checkpoint(s)")


class SimpleChunkedModelWithCheckpoints:
    """Enhanced chunked model with full checkpoint support"""
    
    def __init__(self, model_path: str, max_vram_gb: float = 4.0, temperature: float = 0.7, 
                 quiet: bool = False, chunk_layers: int = None, checkpoint_dir: str = "./checkpoints"):
        self.model_path = model_path
        self.max_vram_gb = max_vram_gb
        self.temperature = temperature
        self.quiet = quiet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Model components
        self.tokenizer = None
        self.config = None
        self.safetensor_files = []
        self.layer_weights = {}
        self.embedding_weights = None
        self.lm_head_weights = None
        
        # Performance tracking
        self.generation_start_time = 0.0
        self.tokens_generated = 0
        self.total_chunks_processed = 0
        self.chunk_times = []
        
        # Auto-detect precision and chunk size
        if max_vram_gb <= 4:
            self.dtype = torch.float16
            default_layers = 1
            self.keep_embeddings_on_cpu = True
        elif max_vram_gb <= 8:
            self.dtype = torch.float16
            default_layers = 2
            self.keep_embeddings_on_cpu = False
        else:
            self.dtype = torch.bfloat16
            default_layers = 4
            self.keep_embeddings_on_cpu = False
        
        self.max_layers_in_vram = chunk_layers if chunk_layers is not None else default_layers
        
        if not self.quiet:
            print(f"🚀 Initializing chunked model with checkpoint support")
            print(f"   Model: {model_path}")
            print(f"   VRAM limit: {max_vram_gb}GB")
            print(f"   Device: {self.device}")
            print(f"   Precision: {self.dtype}")
            print(f"   Max layers per chunk: {self.max_layers_in_vram}")
            print(f"   CPU offloading: {self.keep_embeddings_on_cpu}")
            print(f"   Checkpoint directory: {checkpoint_dir}")
            
            estimated_chunks = 80 // self.max_layers_in_vram
            print(f"   Expected chunks per token: ~{estimated_chunks}")
    
    def get_model_config_hash(self) -> str:
        """Generate hash of model configuration for compatibility checking"""
        config_str = f"{self.model_path}_{self.config.num_hidden_layers}_{self.config.hidden_size}_{self.config.vocab_size}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def save_rng_state(self) -> Dict:
        """Save random number generator states for reproducibility"""
        return {
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng': np.random.get_state(),
        }
    
    def restore_rng_state(self, rng_state: Dict):
        """Restore random number generator states"""
        if rng_state and 'torch_rng' in rng_state:
            torch.set_rng_state(rng_state['torch_rng'])
            if torch.cuda.is_available() and rng_state['torch_cuda_rng'] is not None:
                torch.cuda.set_rng_state(rng_state['torch_cuda_rng'])
            if 'numpy_rng' in rng_state:
                np.random.set_state(rng_state['numpy_rng'])
    
    def create_checkpoint(self, prompt: str, generated_tokens: List[str], 
                         current_token_count: int, target_token_count: int,
                         hidden_states: torch.Tensor) -> GenerationCheckpoint:
        """Create a checkpoint from current generation state"""
        current_time = time.time()
        elapsed_time = current_time - self.generation_start_time
        tokens_per_second = self.tokens_generated / elapsed_time if elapsed_time > 0 else 0.0
        
        remaining_tokens = target_token_count - current_token_count
        estimated_time_remaining = remaining_tokens / tokens_per_second if tokens_per_second > 0 else 0.0
        
        # Get current sequence token IDs
        sequence_text = prompt + "".join(generated_tokens)
        sequence_ids = self.tokenizer.encode(sequence_text)
        
        return GenerationCheckpoint(
            prompt=prompt,
            generated_tokens=generated_tokens.copy(),
            current_token_count=current_token_count,
            target_token_count=target_token_count,
            current_sequence_ids=sequence_ids,
            model_path=self.model_path,
            temperature=self.temperature,
            max_vram_gb=self.max_vram_gb,
            chunk_layers=self.max_layers_in_vram,
            timestamp=current_time,
            model_config_hash=self.get_model_config_hash(),
            torch_rng_state=self.save_rng_state(),
            estimated_time_remaining=estimated_time_remaining,
            tokens_per_second=tokens_per_second,
            total_chunks_processed=self.total_chunks_processed,
            average_chunk_time=np.mean(self.chunk_times) if self.chunk_times else 0.0
        )
    
    # Include all the existing methods from SimpleChunkedModel
    def load_tokenizer_and_config(self):
        """Load tokenizer and config (lightweight)"""
        if not self.quiet:
            print("📝 Loading tokenizer and config...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.config = AutoConfig.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if not self.quiet:
                print(f"   Model type: {self.config.model_type}")
                print(f"   Layers: {self.config.num_hidden_layers}")
                print(f"   Hidden size: {self.config.hidden_size}")
                print(f"   Vocab size: {self.config.vocab_size}")
            
            return True
            
        except Exception as e:
            if not self.quiet:
                print(f"❌ Error loading tokenizer/config: {e}")
            return False
    
    def create_memory_maps(self):
        """Find safetensor files"""
        if not self.quiet:
            print("🗺️  Finding model weight files...")
        
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            from huggingface_hub import snapshot_download
            if not self.quiet:
                print(f"   Downloading {self.model_path}...")
            model_dir = Path(snapshot_download(self.model_path))
        
        self.safetensor_files = list(model_dir.glob("*.safetensors"))
        if not self.safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
        
        if not self.quiet:
            print(f"✅ Found {len(self.safetensor_files)} model files")
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load specific layer weights from safetensor files"""
        layer_weights = {}
        
        layer_patterns = [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight", 
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"model.layers.{layer_idx}.mlp.down_proj.weight",
            f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        ]
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    for pattern in layer_patterns:
                        if pattern in f.keys():
                            tensor = f.get_tensor(pattern)
                            layer_weights[pattern] = tensor.to(device=self.device, dtype=self.dtype)
            except Exception as e:
                if not self.quiet:
                    print(f"   Warning: Could not read {file_path.name}: {e}")
                continue
        
        return layer_weights
    
    def get_embedding_weights(self) -> Optional[torch.Tensor]:
        """Load embedding layer weights"""
        if self.embedding_weights is not None:
            if self.keep_embeddings_on_cpu:
                return self.embedding_weights.to(device=self.device, dtype=self.dtype)
            else:
                return self.embedding_weights
        
        embed_patterns = ["model.embed_tokens.weight", "transformer.wte.weight"]
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    for pattern in embed_patterns:
                        if pattern in f.keys():
                            tensor = f.get_tensor(pattern)
                            if self.keep_embeddings_on_cpu:
                                self.embedding_weights = tensor.to(dtype=self.dtype)
                                return tensor.to(device=self.device, dtype=self.dtype)
                            else:
                                self.embedding_weights = tensor.to(device=self.device, dtype=self.dtype)
                                return self.embedding_weights
            except Exception:
                continue
        return None
    
    def get_lm_head_weights(self) -> Optional[torch.Tensor]:
        """Load language modeling head weights"""
        if self.lm_head_weights is not None:
            if self.keep_embeddings_on_cpu:
                return self.lm_head_weights.to(device=self.device, dtype=self.dtype)
            else:
                return self.lm_head_weights
        
        lm_head_patterns = ["lm_head.weight", "transformer.wte.weight", "model.embed_tokens.weight"]
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    for pattern in lm_head_patterns:
                        if pattern in f.keys():
                            tensor = f.get_tensor(pattern)
                            if self.keep_embeddings_on_cpu:
                                self.lm_head_weights = tensor.to(dtype=self.dtype)
                                return tensor.to(device=self.device, dtype=self.dtype)
                            else:
                                self.lm_head_weights = tensor.to(device=self.device, dtype=self.dtype)
                                return self.lm_head_weights
            except Exception:
                continue
        return None
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def clear_vram_cache(self):
        """Clear VRAM cache and force garbage collection"""
        for layer_idx in list(self.layer_weights.keys()):
            if self.layer_weights[layer_idx]:
                for weight_name in list(self.layer_weights[layer_idx].keys()):
                    del self.layer_weights[layer_idx][weight_name]
            del self.layer_weights[layer_idx]
        
        self.layer_weights.clear()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def load_layer_chunk(self, start_layer: int, end_layer: int, verbose: bool = True):
        """Load a chunk of layers into VRAM"""
        chunk_start_time = time.time()
        
        self.clear_vram_cache()
        loaded_layers = 0
        
        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx in self.layer_weights:
                loaded_layers += 1
                continue
            
            current_vram = self.get_vram_usage()
            if current_vram > self.max_vram_gb * 0.7:
                break
            
            try:
                layer_weights = self.get_layer_weights(layer_idx)
                if layer_weights:
                    self.layer_weights[layer_idx] = layer_weights
                    loaded_layers += 1
                    
                    new_vram = self.get_vram_usage()
                    if new_vram > self.max_vram_gb * 0.8:
                        break
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.clear_vram_cache()
                    break
        
        chunk_time = time.time() - chunk_start_time
        self.chunk_times.append(chunk_time)
        self.total_chunks_processed += 1
        
        return loaded_layers > 0
    
    def simple_attention(self, hidden_states: torch.Tensor, layer_idx: int, verbose: bool = True) -> torch.Tensor:
        """Grouped Query Attention computation"""
        if layer_idx not in self.layer_weights:
            return hidden_states
        
        weights = self.layer_weights[layer_idx]
        
        q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        
        if not all(key in weights for key in [q_key, k_key, v_key, o_key]):
            return hidden_states
        
        try:
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            q_weight = weights[q_key]
            k_weight = weights[k_key]
            v_weight = weights[v_key]
            
            q_size = q_weight.shape[0]
            k_size = k_weight.shape[0]
            
            num_q_heads = getattr(self.config, 'num_attention_heads', 64)
            head_dim = hidden_size // num_q_heads
            num_kv_heads = k_size // head_dim
            
            q = F.linear(hidden_states, q_weight)
            k = F.linear(hidden_states, k_weight)
            v = F.linear(hidden_states, v_weight)
            
            q = q.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            if num_kv_heads < num_q_heads:
                repeat_factor = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            
            attention_scores = torch.matmul(q, k.transpose(-2, -1))
            attention_scores = attention_scores / (head_dim ** 0.5)
            attention_probs = F.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_probs, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            output = F.linear(context, weights[o_key])
            
            return output
            
        except Exception as e:
            if verbose:
                print(f"   Error in attention for layer {layer_idx}: {e}")
            return hidden_states
    
    def simple_mlp(self, hidden_states: torch.Tensor, layer_idx: int, verbose: bool = True) -> torch.Tensor:
        """MLP computation"""
        if layer_idx not in self.layer_weights:
            return hidden_states
        
        weights = self.layer_weights[layer_idx]
        
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        
        if not all(key in weights for key in [gate_key, up_key, down_key]):
            return hidden_states
        
        try:
            gate = F.linear(hidden_states, weights[gate_key])
            up = F.linear(hidden_states, weights[up_key])
            gate = gate * torch.sigmoid(gate)
            intermediate = gate * up
            output = F.linear(intermediate, weights[down_key])
            return output
        except Exception as e:
            if verbose:
                print(f"   Error in MLP for layer {layer_idx}: {e}")
            return hidden_states
    
    def process_layer_chunk(self, hidden_states: torch.Tensor, start_layer: int, end_layer: int, verbose: bool = True) -> torch.Tensor:
        """Process hidden states through a chunk of layers"""
        current_states = hidden_states
        
        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx not in self.layer_weights:
                continue
            
            residual = current_states
            
            # Layer norm
            ln_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            if ln_key in self.layer_weights[layer_idx]:
                try:
                    ln_weight = self.layer_weights[layer_idx][ln_key]
                    current_states = F.layer_norm(current_states, (current_states.size(-1),), ln_weight)
                except Exception:
                    pass
            
            # Attention
            attn_output = self.simple_attention(current_states, layer_idx, verbose=verbose)
            current_states = residual + attn_output
            
            # Post attention layer norm and MLP
            residual = current_states
            post_ln_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            if post_ln_key in self.layer_weights[layer_idx]:
                try:
                    post_ln_weight = self.layer_weights[layer_idx][post_ln_key]
                    current_states = F.layer_norm(current_states, (current_states.size(-1),), post_ln_weight)
                except Exception:
                    pass
            
            # MLP
            mlp_output = self.simple_mlp(current_states, layer_idx, verbose=verbose)
            current_states = residual + mlp_output
        
        return current_states
    
    def generate_chunked_with_checkpoints(self, prompt: str = None, max_tokens: int = 50,
                                        checkpoint_every: int = 10, checkpoint_name: str = None,
                                        resume_from: str = None) -> Generator[str, None, None]:
        """Generate text with checkpoint support"""
        
        self.generation_start_time = time.time()
        self.tokens_generated = 0
        self.total_chunks_processed = 0
        self.chunk_times = []
        
        # Handle resume logic
        if resume_from:
            checkpoint, hidden_states = self.checkpoint_manager.load_checkpoint(resume_from)
            
            # Validate checkpoint
            if not self.checkpoint_manager.validate_checkpoint(checkpoint, self.get_model_config_hash()):
                raise ValueError("Checkpoint validation failed")
            
            # Restore state
            prompt = checkpoint.prompt
            generated_tokens = checkpoint.generated_tokens
            start_token = checkpoint.current_token_count
            max_tokens = checkpoint.target_token_count  # Use original target
            hidden_states = hidden_states.to(self.device)
            
            # Restore RNG state for reproducibility
            if checkpoint.torch_rng_state:
                self.restore_rng_state(checkpoint.torch_rng_state)
            
            # Yield already generated tokens for display
            if not self.quiet:
                print(f"🔄 Resuming generation from token {start_token}")
                print("🤖 Previously generated:", end="")
            
            for token in generated_tokens:
                if not self.quiet:
                    print(f" {token}", end="", flush=True)
                yield token
            
            if not self.quiet:
                print(f"\n🎯 Continuing generation...")
        
        else:
            # Normal initialization
            if not self.quiet:
                print(f"🎯 Starting new generation: '{prompt[:50]}...'")
            
            # Tokenize and embed
            #inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > 4000:  # Use model's actual limit
                print(f"⚠️  Input too long, truncating to 4000 tokens")
                inputs["input_ids"] = inputs["input_ids"][:, -4000:]
            
            input_ids = inputs["input_ids"]
            
            embed_weights = self.get_embedding_weights()
            if embed_weights is None:
                raise RuntimeError("Could not load embedding weights")
            
            hidden_states = F.embedding(input_ids, embed_weights).to(self.dtype)
            
            if self.keep_embeddings_on_cpu:
                del embed_weights
                torch.cuda.empty_cache()
            
            generated_tokens = []
            start_token = 0
        
        total_layers = self.config.num_hidden_layers
        
        # Main generation loop with checkpoints
        for token_idx in range(start_token, max_tokens):
            # Move hidden states to CPU during layer processing
            if self.keep_embeddings_on_cpu:
                hidden_states_cpu = hidden_states.cpu()
                del hidden_states
                torch.cuda.empty_cache()
            else:
                hidden_states_cpu = hidden_states
            
            current_states = hidden_states_cpu
            total_chunks = (total_layers + self.max_layers_in_vram - 1) // self.max_layers_in_vram
            
            # Process through all layers in chunks
            for chunk_idx, start_layer in enumerate(range(0, total_layers, self.max_layers_in_vram)):
                end_layer = min(start_layer + self.max_layers_in_vram - 1, total_layers - 1)
                
                # Progress indicator
                progress = int((chunk_idx / total_chunks) * 100)
                dots = "." * (progress // 5)
                print(f"\rToken {token_idx + 1}/{max_tokens}: Processing ({progress:3d}%) {dots:<20} [Chunk {chunk_idx + 1}/{total_chunks}]", end="", flush=True)
                
                # Move states to GPU for processing
                if self.keep_embeddings_on_cpu and current_states.device == torch.device('cpu'):
                    current_states = current_states.to(self.device)
                
                # Load and process chunk
                success = self.load_layer_chunk(start_layer, end_layer, verbose=False)
                if success and self.layer_weights:
                    current_states = self.process_layer_chunk(current_states, start_layer, end_layer, verbose=False)
                
                self.clear_vram_cache()
                
                # Move states back to CPU between chunks
                if self.keep_embeddings_on_cpu and start_layer < total_layers - self.max_layers_in_vram:
                    current_states = current_states.cpu()
                    torch.cuda.empty_cache()
            
            # Complete progress
            print(f"\rToken {token_idx + 1}/{max_tokens}: Processing (100%) .................... [Chunk {total_chunks}/{total_chunks}] ✓", end="", flush=True)
            
            # Ensure final states are on GPU for token generation
            if current_states.device == torch.device('cpu'):
                current_states = current_states.to(self.device)
            
            hidden_states = current_states
            
            # Generate next token
            lm_head_weights = self.get_lm_head_weights()
            if lm_head_weights is None:
                lm_head_weights = self.get_embedding_weights()
            
            if lm_head_weights is not None:
                last_hidden = hidden_states[:, -1, :]
                logits = F.linear(last_hidden, lm_head_weights)
                
                if self.keep_embeddings_on_cpu:
                    del lm_head_weights
                    torch.cuda.empty_cache()
                
                logits = logits / self.temperature
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                next_token_id = torch.randint(0, self.config.vocab_size, (1, 1), device=self.device)
            
            # Decode token
            next_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(f" {next_token}", end="", flush=True)
            generated_tokens.append(next_token)
            yield next_token
            
            # Update hidden states
            embed_weights = self.get_embedding_weights()
            if embed_weights is not None:
                new_embedding = F.embedding(next_token_id, embed_weights).to(self.dtype)
                if self.keep_embeddings_on_cpu:
                    del embed_weights
                    torch.cuda.empty_cache()
                hidden_states = torch.cat([hidden_states, new_embedding], dim=1)
            
            self.tokens_generated += 1
            
            # Save checkpoint periodically
            if checkpoint_name and (token_idx + 1) % checkpoint_every == 0:
                checkpoint = self.create_checkpoint(prompt, generated_tokens, token_idx + 1, max_tokens, hidden_states)
                self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name, hidden_states, async_save=True)
            
            # Early stopping
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # Final checkpoint
        if checkpoint_name:
            final_checkpoint = self.create_checkpoint(prompt, generated_tokens, len(generated_tokens), max_tokens, hidden_states)
            self.checkpoint_manager.save_checkpoint(final_checkpoint, checkpoint_name + "_final", hidden_states, async_save=False)
    
    def cleanup(self):
        """Clean up resources"""
        if not self.quiet:
            print("\n🧹 Cleaning up...")
        
        self.layer_weights.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if not self.quiet:
            print("✅ Cleanup complete")


def main():
    """Main application with checkpoint support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunked Inference with Checkpoint Support")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace model name")
    parser.add_argument("--vram", type=float, default=4.0, help="Max VRAM in GB")
    parser.add_argument("--prompt", default="The future of AI is", help="Generation prompt")
    parser.add_argument("--tokens", type=int, default=20, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--quiet", action="store_true", help="Hide initialization messages")
    parser.add_argument("--chunk-layers", type=int, help="Layers per chunk")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N tokens")
    parser.add_argument("--checkpoint-name", help="Base name for checkpoint files")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-from", help="Resume from checkpoint file")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    parser.add_argument("--cleanup-checkpoints", action="store_true", help="Remove old checkpoint files")
    
    args = parser.parse_args()
    
    # Handle checkpoint management commands
    if args.list_checkpoints:
        manager = CheckpointManager(args.checkpoint_dir)
        manager.list_checkpoints()
        return
    
    if args.cleanup_checkpoints:
        manager = CheckpointManager(args.checkpoint_dir)
        manager.cleanup_old_checkpoints()
        return
    
    # Create model
    model = SimpleChunkedModelWithCheckpoints(
        args.model, args.vram, args.temperature, args.quiet, 
        args.chunk_layers, args.checkpoint_dir
    )
    
    try:
        # Initialize
        if not model.load_tokenizer_and_config():
            print("❌ Failed to load tokenizer/config")
            return
        
        model.create_memory_maps()
        
        # Generate with checkpoints
        if not args.quiet:
            if args.resume_from:
                print(f"\n🔄 Resuming from: {args.resume_from}")
            else:
                print(f"\n🎯 Prompt: {args.prompt}")
            print("🤖 Response:", end="", flush=True)
        
        for token in model.generate_chunked_with_checkpoints(
            prompt=args.prompt,
            max_tokens=args.tokens,
            checkpoint_every=args.checkpoint_every,
            checkpoint_name=args.checkpoint_name,
            resume_from=args.resume_from
        ):
            pass  # Token is already printed in generate method
        
        print(f"\n{'✅ Generation complete!' if not args.quiet else ''}")
        
        # Show final stats
        if not args.quiet:
            elapsed_time = time.time() - model.generation_start_time
            tokens_per_second = model.tokens_generated / elapsed_time if elapsed_time > 0 else 0
            print(f"📊 Performance: {model.tokens_generated} tokens in {elapsed_time:.1f}s ({tokens_per_second:.2f} tok/s)")
            print(f"🔧 Processed {model.total_chunks_processed} chunks total")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Generation interrupted. Progress saved in checkpoints.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.cleanup()


if __name__ == "__main__":
    main()
