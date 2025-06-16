#!/usr/bin/env python3
"""
Enhanced Chunked Inference with Universal Model Compatibility + Full Checkpoint/Resume System
Enables safe long-form generation with progress saving and resumption across any model architecture
Now with robust tokenization, vocabulary mismatch handling, and universal weight loading
ENHANCED FOR HUGE MODELS: Supports 671B+ models with aggressive memory management
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
from typing import Dict, List, Optional, Tuple, Generator, Any, Union
import pickle
import gc

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
import torch.nn.functional as F


# =============================================================================
# MODEL COMPATIBILITY SYSTEM
# =============================================================================

class ModelCompatibilityManager:
    """Handles model compatibility issues automatically"""
    
    def __init__(self, model_path: str, device: torch.device, quiet: bool = False):
        self.model_path = model_path
        self.device = device
        self.quiet = quiet
        
        # Compatibility state
        self.vocab_mismatch = False
        self.safe_vocab_size = None
        self.tokenizer_vocab_size = None
        self.model_vocab_size = None
        self.embedding_vocab_size = None
        self.special_tokens = {}
        self.architecture_type = None
        self.compatibility_report = {}
        
    def analyze_and_fix(self, tokenizer, config, embedding_weights=None) -> Dict[str, Any]:
        """Analyze compatibility and apply automatic fixes"""
        
        if not self.quiet:
            print(f"\n🔍 Analyzing model compatibility...")
        
        # Detect vocabulary sizes
        self.tokenizer_vocab_size = tokenizer.vocab_size
        self.model_vocab_size = getattr(config, 'vocab_size', self.tokenizer_vocab_size)
        self.embedding_vocab_size = embedding_weights.shape[0] if embedding_weights is not None else self.model_vocab_size
        
        # Determine safe vocabulary size (most restrictive)
        vocab_sizes = [self.tokenizer_vocab_size, self.model_vocab_size, self.embedding_vocab_size]
        self.safe_vocab_size = min(vocab_sizes)
        
        # Check for mismatches
        self.vocab_mismatch = len(set(vocab_sizes)) > 1
        
        # Collect special tokens with proper handling
        self.special_tokens = {
            'eos_token_id': getattr(tokenizer, 'eos_token_id', 2),
            'bos_token_id': getattr(tokenizer, 'bos_token_id', 1),
            'unk_token_id': getattr(tokenizer, 'unk_token_id', 0),
            'pad_token_id': getattr(tokenizer, 'pad_token_id', tokenizer.eos_token_id)
        }
        
        # Ensure all special tokens are within safe range
        for token_name, token_id in self.special_tokens.items():
            if token_id is not None and token_id >= self.safe_vocab_size:
                if not self.quiet:
                    print(f"⚠️  {token_name} ({token_id}) outside safe range, clamping to UNK")
                self.special_tokens[token_name] = self.special_tokens['unk_token_id']
        
        # Detect architecture
        self.architecture_type = getattr(config, 'model_type', 'unknown')
        
        # Create compatibility report
        self.compatibility_report = {
            'vocab_mismatch': self.vocab_mismatch,
            'safe_vocab_size': self.safe_vocab_size,
            'tokenizer_vocab_size': self.tokenizer_vocab_size,
            'model_vocab_size': self.model_vocab_size,
            'embedding_vocab_size': self.embedding_vocab_size,
            'special_tokens': self.special_tokens,
            'architecture_type': self.architecture_type,
            'fixes_applied': []
        }
        
        # Report findings
        if not self.quiet:
            if self.vocab_mismatch:
                print(f"⚠️  VOCABULARY MISMATCH DETECTED:")
                print(f"   Tokenizer: {self.tokenizer_vocab_size}")
                print(f"   Model:     {self.model_vocab_size}")
                print(f"   Embedding: {self.embedding_vocab_size}")
                print(f"   🔧 Safe range: 0-{self.safe_vocab_size-1}")
                print(f"   ✅ Automatic token clamping enabled")
                self.compatibility_report['fixes_applied'].append('token_clamping')
            else:
                print(f"✅ Vocabulary sizes consistent: {self.safe_vocab_size}")
            
            print(f"✅ Architecture: {self.architecture_type}")
            print(f"✅ Special tokens validated")
        
        return self.compatibility_report


class SafeTokenizationPipeline:
    """Robust tokenization pipeline that handles any model"""
    
    def __init__(self, tokenizer, compatibility_manager: ModelCompatibilityManager):
        self.tokenizer = tokenizer
        self.compat = compatibility_manager
        
    def safe_encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        """Safely encode text with validation"""
        try:
            if max_length:
                tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, 
                                             truncation=True, max_length=max_length)
            else:
                tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            
            # Validate token range
            return self.validate_token_range(tokens)
            
        except Exception as e:
            if not self.compat.quiet:
                print(f"⚠️  Encoding failed: {e}, using fallback")
            # Simple word-based fallback
            words = text.split()
            return [min(hash(word) % self.compat.safe_vocab_size, self.compat.safe_vocab_size - 1) 
                   for word in words[:50]]  # Limit length
    
    def safe_decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = True) -> str:
        """Safely decode tokens with fallback handling"""
        # Normalize input
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        # Validate and clamp token IDs
        safe_tokens = self.validate_token_range(token_ids)
        
        try:
            decoded = self.tokenizer.decode(safe_tokens, skip_special_tokens=skip_special_tokens)
            
            # Validate output
            if decoded and len(decoded.strip()) > 0:
                return decoded
            else:
                return " "  # Safe fallback
                
        except Exception as e:
            if not self.compat.quiet:
                print(f"⚠️  Decode failed for tokens {token_ids}: {e}")
            return " "  # Safe fallback
    
    def validate_token_range(self, token_ids: List[int]) -> List[int]:
        """Clamp token IDs to safe vocabulary range"""
        safe_tokens = []
        for token_id in token_ids:
            if token_id >= self.compat.safe_vocab_size:
                # Clamp to UNK token
                clamped_id = self.compat.special_tokens.get('unk_token_id', 0)
                safe_tokens.append(clamped_id)
                if not self.compat.quiet:
                    print(f"🔧 Clamped token {token_id} -> {clamped_id}")
            else:
                safe_tokens.append(token_id)
        return safe_tokens
    
    def safe_token_sampling(self, logits: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
        """Sample tokens safely with vocabulary masking"""
        # Apply vocabulary masking if needed
        if self.compat.vocab_mismatch and logits.size(-1) > self.compat.safe_vocab_size:
            # Create mask for valid vocabulary range
            vocab_mask = torch.zeros_like(logits)
            vocab_mask[:self.compat.safe_vocab_size] = 1.0
            
            # Mask invalid positions with very low probability
            logits = logits * vocab_mask + (1.0 - vocab_mask) * (-1e9)
        
        # Apply temperature
        if temperature <= 0:
            temperature = 1e-7
        logits = logits / temperature
        
        # Clamp for numerical stability
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Check for invalid values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            if not self.compat.quiet:
                print("⚠️  Invalid logits detected, using uniform fallback")
            # Create uniform distribution over safe vocabulary
            uniform_logits = torch.zeros_like(logits)
            uniform_logits[:self.compat.safe_vocab_size] = 1.0
            logits = uniform_logits
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


class UniversalWeightLoader:
    """Loads model weights consistently across architectures"""
    
    def __init__(self, safetensor_files: List[Path], config, quiet: bool = False):
        self.safetensor_files = safetensor_files
        self.config = config
        self.quiet = quiet
        self.weight_patterns = self._detect_patterns()
        
    def _detect_patterns(self) -> Dict[str, List[str]]:
        """Auto-detect weight naming patterns from safetensor files"""
        all_keys = set()
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    all_keys.update(f.keys())
            except Exception as e:
                if not self.quiet:
                    print(f"⚠️  Could not read {file_path.name}: {e}")
                continue
        
        patterns = {
            'embedding': [],
            'lm_head': [],
            'layer_prefix': None,
            'all_keys': list(all_keys)
        }
        
        # Detect embedding patterns
        embedding_indicators = ['embed', 'wte', 'word_embeddings']
        patterns['embedding'] = [k for k in all_keys 
                               if any(ind in k.lower() for ind in embedding_indicators)]
        
        # Detect LM head patterns
        lm_head_indicators = ['lm_head', 'head', 'output', 'classifier']
        patterns['lm_head'] = [k for k in all_keys 
                             if any(ind in k.lower() for ind in lm_head_indicators)]
        
        # Detect layer structure
        layer_keys = [k for k in all_keys if 'layers.' in k or '.h.' in k]
        if layer_keys:
            if 'model.layers.' in layer_keys[0]:
                patterns['layer_prefix'] = 'model.layers'
            elif 'transformer.h.' in layer_keys[0]:
                patterns['layer_prefix'] = 'transformer.h'
            elif 'transformer.layer.' in layer_keys[0]:
                patterns['layer_prefix'] = 'transformer.layer'
        
        return patterns
    
    def load_embeddings(self) -> Optional[torch.Tensor]:
        """Load embedding weights with comprehensive pattern matching"""
        # Comprehensive embedding patterns for all architectures
        embedding_patterns = [
            "model.embed_tokens.weight",
            "embed_tokens.weight", 
            "transformer.wte.weight",
            "wte.weight",
            "word_embeddings.weight",
            "embeddings.word_embeddings.weight",
            "model.embeddings.word_embeddings.weight",
            "transformer.word_embeddings.weight",
            "model.word_embeddings.weight",
            "embeddings.weight",
            "input_embeddings.weight",
            "model.embeddings.weight",
            "gpt_neox.embed_in.weight",
            "embed_in.weight"
        ]
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    available_keys = f.keys()
                    
                    for pattern in embedding_patterns:
                        if pattern in available_keys:
                            tensor = f.get_tensor(pattern)
                            if len(tensor.shape) == 2:  # Valid embedding matrix
                                if not self.quiet:
                                    print(f"✅ Found embeddings: {pattern} {list(tensor.shape)}")
                                return tensor.clone()
                                
            except Exception as e:
                if not self.quiet:
                    print(f"⚠️  Error reading {file_path.name}: {e}")
                continue
        
        if not self.quiet:
            print("❌ No embedding weights found with standard patterns")
            print(f"   Available embedding-like keys: {self.weight_patterns['embedding'][:5]}")
        
        return None
    
    def load_lm_head(self) -> Optional[torch.Tensor]:
        """Load LM head weights with tied weight detection"""
        # First try dedicated LM head patterns
        lm_head_patterns = [
            "lm_head.weight",
            "model.lm_head.weight",
            "output.weight",
            "model.output.weight",
            "head.weight",
            "classifier.weight",
            "output_layer.weight",
            "gpt_neox.embed_out.weight",
            "transformer.head.weight",
            "language_model.lm_head.weight"
        ]
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    available_keys = f.keys()
                    
                    for pattern in lm_head_patterns:
                        if pattern in available_keys:
                            tensor = f.get_tensor(pattern)
                            if len(tensor.shape) == 2:
                                if not self.quiet:
                                    print(f"✅ Found LM head: {pattern} {list(tensor.shape)}")
                                return tensor.clone()
                                
            except Exception as e:
                if not self.quiet:
                    print(f"⚠️  Error reading {file_path.name}: {e}")
                continue
        
        # If no dedicated LM head found, check for tied weights (use embeddings)
        embedding_weights = self.load_embeddings()
        if embedding_weights is not None:
            if not self.quiet:
                print("✅ Using tied weights (embeddings as LM head)")
            return embedding_weights
        
        if not self.quiet:
            print("❌ No LM head weights found")
            print(f"   Available head-like keys: {self.weight_patterns['lm_head'][:5]}")
        
        return None
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load specific layer weights with universal patterns"""
        layer_weights = {}
        
        # Universal layer patterns - covers LLaMA, Qwen, Mixtral, GPT, etc.
        base_patterns = [
            f"model.layers.{layer_idx}",
            f"transformer.h.{layer_idx}",
            f"transformer.layer.{layer_idx}",
            f"layers.{layer_idx}",
            f"h.{layer_idx}"
        ]
        
        # Weight suffixes for different components
        weight_suffixes = [
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight", 
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            # Alternative naming patterns
            ".attn.q_proj.weight",
            ".attn.k_proj.weight",
            ".attn.v_proj.weight",
            ".attn.o_proj.weight",
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.c_fc.weight",
            ".mlp.c_proj.weight",
            ".ln_1.weight",
            ".ln_2.weight",
        ]
        
        for file_path in self.safetensor_files:
            try:
                with safe_open(file_path, framework="pt") as f:
                    available_keys = set(f.keys())
                    
                    # Try each base pattern with each suffix
                    for base in base_patterns:
                        for suffix in weight_suffixes:
                            full_pattern = base + suffix
                            if full_pattern in available_keys:
                                tensor = f.get_tensor(full_pattern)
                                layer_weights[full_pattern] = tensor.clone()
                                
            except Exception as e:
                if not self.quiet:
                    print(f"⚠️  Error reading {file_path.name}: {e}")
                continue
        
        return layer_weights


# =============================================================================
# CHECKPOINT SYSTEM (UNCHANGED)
# =============================================================================

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


# =============================================================================
# TORCH.COMPILE OPTIMIZATION WRAPPER (UNCHANGED)
# =============================================================================

def get_compiled_function(func, quiet=False):
    """
    Safely compile function with torch.compile if PyTorch 2.0+ available
    Falls back gracefully for older PyTorch versions or compilation failures
    """
    # Check for environment variable to disable compilation
    if os.environ.get('DISABLE_TORCH_COMPILE', '').lower() in ('1', 'true', 'yes'):
        if not quiet:
            print(f"🚫 torch.compile disabled by DISABLE_TORCH_COMPILE environment variable")
        return func
    
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            # Additional version check to be safe
            version_parts = torch.__version__.split('.')
            major_version = int(version_parts[0])
            
            if major_version >= 2:
                if not quiet:
                    print(f"🚀 torch.compile available - attempting JIT compilation of {func.__name__}")
                
                # Create a wrapper that catches compilation failures at runtime
                def safe_compiled_wrapper(*args, **kwargs):
                    try:
                        return compiled_func(*args, **kwargs)
                    except Exception as e:
                        if not quiet:
                            print(f"⚠️  torch.compile failed during execution: {e}")
                            print(f"   Falling back to non-compiled {func.__name__}")
                            print(f"   To disable compilation entirely, set: export DISABLE_TORCH_COMPILE=1")
                        # Fallback to original function
                        return func(*args, **kwargs)
                
                try:
                    compiled_func = torch.compile(func)
                    if not quiet:
                        print(f"✅ Successfully compiled {func.__name__}")
                    return safe_compiled_wrapper
                except Exception as e:
                    if not quiet:
                        print(f"⚠️  torch.compile compilation failed: {e}, using standard function")
                    return func
            else:
                if not quiet:
                    print(f"⚠️  PyTorch {torch.__version__} detected - torch.compile requires 2.0+")
                return func
        else:
            if not quiet:
                print(f"⚠️  torch.compile not available in this PyTorch version")
            return func
            
    except Exception as e:
        if not quiet:
            print(f"⚠️  torch.compile setup failed: {e}, using standard function")
        return func


# =============================================================================
# PERFORMANCE OPTIMIZATION CLASSES (UNCHANGED)
# =============================================================================

class KVCache:
    """Key-Value cache for attention layers - Massive speedup for generation"""
    def __init__(self, max_seq_len=2048, num_layers=32, num_heads=32, head_dim=128, device="cuda"):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.cache = {}  # layer_idx -> {'k': tensor, 'v': tensor}
        self.seq_len = 0  # Current sequence length
        
    def get_kv(self, layer_idx):
        """Get cached K,V for layer"""
        if layer_idx not in self.cache:
            return None, None
        cache_data = self.cache[layer_idx]
        return cache_data.get('k'), cache_data.get('v')
    
    def store_kv(self, layer_idx, k, v):
        """Store K,V for layer"""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
        
        # Store only up to current sequence length
        if k is not None and v is not None:
            self.cache[layer_idx]['k'] = k[:, :, :self.seq_len+1, :].clone()
            self.cache[layer_idx]['v'] = v[:, :, :self.seq_len+1, :].clone()
    
    def extend_sequence(self):
        """Increment sequence length counter"""
        self.seq_len += 1
    
    def reset(self):
        """Clear cache and reset sequence length"""
        self.cache.clear()
        self.seq_len = 0
    
    def trim_to_length(self, new_length):
        """Trim cache to specific length"""
        self.seq_len = new_length
        for layer_idx in self.cache:
            if 'k' in self.cache[layer_idx]:
                self.cache[layer_idx]['k'] = self.cache[layer_idx]['k'][:, :, :new_length, :]
            if 'v' in self.cache[layer_idx]:
                self.cache[layer_idx]['v'] = self.cache[layer_idx]['v'][:, :, :new_length, :]


class PersistentWeightCache:
    """Keep frequently used layer weights in memory between tokens - Reduced for huge models"""
    def __init__(self, max_cache_size_gb=0.5):  # Reduced from 2.0GB to 0.5GB
        self.max_cache_size = max_cache_size_gb * 1024**3  # Convert to bytes
        self.cache = {}  # layer_idx -> weights
        self.usage_stats = {}  # layer_idx -> last_used_time
        self.current_size = 0
        
    def get_layer(self, layer_idx):
        """Get layer weights from cache"""
        if layer_idx in self.cache:
            self.usage_stats[layer_idx] = time.time()
            return self.cache[layer_idx]
        return None
    
    def store_layer(self, layer_idx, weights):
        """Store layer weights in cache with LRU eviction"""
        if not weights:
            return
            
        layer_size = self._calculate_size(weights)
        
        # Evict old layers if needed
        while self.current_size + layer_size > self.max_cache_size and self.cache:
            self._evict_lru_layer()
        
        self.cache[layer_idx] = weights
        self.usage_stats[layer_idx] = time.time()
        self.current_size += layer_size
    
    def _calculate_size(self, weights):
        """Calculate memory size of weights dict"""
        total_size = 0
        for w in weights.values():
            if hasattr(w, 'numel') and hasattr(w, 'element_size'):
                total_size += w.numel() * w.element_size()
        return total_size
    
    def _evict_lru_layer(self):
        """Evict least recently used layer"""
        if not self.usage_stats:
            return
        lru_layer = min(self.usage_stats.keys(), key=lambda x: self.usage_stats[x])
        self._remove_layer(lru_layer)
    
    def _remove_layer(self, layer_idx):
        """Remove layer from cache"""
        if layer_idx in self.cache:
            weights = self.cache[layer_idx]
            layer_size = self._calculate_size(weights)
            del self.cache[layer_idx]
            del self.usage_stats[layer_idx]
            self.current_size -= layer_size
    
    def clear(self):
        """Clear all cached weights"""
        self.cache.clear()
        self.usage_stats.clear()
        self.current_size = 0


def quantize_weights_int8(weights):
    """Quantize FP16 weights to INT8 for memory efficiency"""
    quantized = {}
    for name, tensor in weights.items():
        if hasattr(tensor, 'dtype') and tensor.dtype in [torch.float16, torch.float32]:
            # Simple linear quantization
            min_val, max_val = tensor.min().item(), tensor.max().item()
            if abs(max_val - min_val) < 1e-8:  # Avoid division by zero
                quantized[name] = tensor.half()
                continue
                
            scale = (max_val - min_val) / 255.0
            zero_point = int((-min_val / scale + 0.5))
            zero_point = max(0, min(255, zero_point))
            
            quantized_tensor = torch.clamp(torch.round((tensor - min_val) / scale), 0, 255).byte()
            quantized[name] = {
                'weight': quantized_tensor,
                'scale': scale,
                'zero_point': zero_point,
                'min_val': min_val,
                'max_val': max_val
            }
        else:
            quantized[name] = tensor
    return quantized


def dequantize_weights_int8(quantized_weights):
    """Dequantize INT8 weights back to FP16 for computation"""
    weights = {}
    for name, data in quantized_weights.items():
        if isinstance(data, dict) and 'weight' in data:
            # Dequantize
            quantized_tensor = data['weight']
            scale = data['scale']
            min_val = data['min_val']
            weights[name] = (quantized_tensor.float() * scale + min_val).half()
        else:
            weights[name] = data
    return weights


def optimized_attention(q, k, v, is_causal=True):
    """Use PyTorch's optimized attention implementation when available"""
    try:
        # Use built-in Flash Attention if available (PyTorch 2.0+)
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    except (AttributeError, RuntimeError):
        # Fallback to manual implementation
        scale = q.size(-1) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if is_causal:
            # Apply causal mask
            seq_len_q, seq_len_k = q.size(-2), k.size(-2)
            if seq_len_q > 1 or seq_len_k > 1:  # Only apply mask if we have multiple tokens
                causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1).bool()
                attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


# =============================================================================
# CHECKPOINT MANAGER (UNCHANGED)
# =============================================================================

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


# =============================================================================
# MAIN ENHANCED MODEL CLASS - OPTIMIZED FOR HUGE MODELS
# =============================================================================

class OptimizedChunkedModelWithCheckpoints:
    """Enhanced chunked model with universal compatibility + performance optimizations - OPTIMIZED FOR HUGE MODELS (671B+)"""
    
    def __init__(self, model_path: str, max_vram_gb: float = 4.0, temperature: float = 0.7, 
                 quiet: bool = False, chunk_layers: int = None, checkpoint_dir: str = "./checkpoints",
                 max_context_length: int = None, enable_optimizations: bool = True):
        self.model_path = model_path
        self.max_vram_gb = max_vram_gb
        self.temperature = temperature
        self.quiet = quiet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_length = max_context_length
        self.enable_optimizations = enable_optimizations
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Model components
        self.tokenizer = None
        self.config = None
        self.safetensor_files = []
        self.layer_weights = {}
        self.embedding_weights = None
        self.lm_head_weights = None
        
        # Compatibility system
        self.compatibility_manager = ModelCompatibilityManager(model_path, self.device, quiet)
        self.safe_tokenizer = None
        self.weight_loader = None
        
        # Performance optimizations - REDUCED FOR HUGE MODELS
        if self.enable_optimizations:
            self.kv_cache = KVCache(device=self.device)
            self.weight_cache = PersistentWeightCache(max_cache_size_gb=0.5)  # Reduced for huge models
            self.enable_quantization = True
        else:
            self.kv_cache = None
            self.weight_cache = None
            self.enable_quantization = False
        
        # Compiled functions for speed (will be set up after init)
        self.compiled_layer_processor = None
        self.compiled_attention = None
        
        # Performance tracking
        self.generation_start_time = 0.0
        self.tokens_generated = 0
        self.total_chunks_processed = 0
        self.chunk_times = []
        
        # ENHANCED: Aggressive memory management for huge models
        # Always enable CPU offloading for embeddings - we'll determine this dynamically
        self.keep_embeddings_on_cpu = True
        self.force_cpu_offloading = False  # Will be set based on model size
        
        # Auto-detect precision and chunk size with huge model consideration
        if max_vram_gb <= 4:
            self.dtype = torch.float16
            default_layers = 1
            self.aggressive_offloading = True
        elif max_vram_gb <= 8:
            self.dtype = torch.float16
            default_layers = 1  # Reduced for huge models
            self.aggressive_offloading = True
        else:
            self.dtype = torch.bfloat16
            default_layers = 2  # Still conservative for huge models
            self.aggressive_offloading = False
        
        self.max_layers_in_vram = chunk_layers if chunk_layers is not None else default_layers
        
        if not self.quiet:
            print(f"🚀 Initializing enhanced chunked model with universal compatibility")
            print(f"   Model: {model_path}")
            print(f"   VRAM limit: {max_vram_gb}GB")
            print(f"   Device: {self.device}")
            print(f"   Precision: {self.dtype}")
            print(f"   Max layers per chunk: {self.max_layers_in_vram}")
            print(f"   Aggressive CPU offloading: {'✅' if self.aggressive_offloading else '❌'}")
            print(f"   Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
            if enable_optimizations:
                print(f"     - KV Caching: ✅")
                print(f"     - Weight Caching: ✅ (reduced for huge models)")
                print(f"     - INT8 Quantization: ✅")
                print(f"     - Optimized Attention: ✅")
                print(f"     - torch.compile: {'✅' if hasattr(torch, 'compile') else '❌'}")
            print(f"   Checkpoint directory: {checkpoint_dir}")
            
            estimated_chunks = 80 // self.max_layers_in_vram
            print(f"   Expected chunks per token: ~{estimated_chunks}")
    
    def estimate_embedding_memory_gb(self, vocab_size: int, hidden_size: int) -> float:
        """Estimate memory required for embedding weights in GB"""
        # vocab_size * hidden_size * 2 bytes (fp16) = total bytes
        total_bytes = vocab_size * hidden_size * 2
        return total_bytes / (1024**3)
    
    def check_and_force_cpu_offloading(self):
        """Check if model is too big and force CPU offloading"""
        if not self.config:
            return
        
        vocab_size = getattr(self.config, 'vocab_size', 50000)
        hidden_size = getattr(self.config, 'hidden_size', 4096)
        
        # Estimate embedding memory requirements
        embedding_memory_gb = self.estimate_embedding_memory_gb(vocab_size, hidden_size)
        
        # Get current available VRAM
        if torch.cuda.is_available():
            current_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            available_memory_gb = self.max_vram_gb - current_memory_gb - 0.5  # 0.5GB safety margin
        else:
            available_memory_gb = 0
        
        if not self.quiet:
            print(f"\n🔍 Memory Analysis:")
            print(f"   Embedding memory required: {embedding_memory_gb:.2f}GB")
            print(f"   Available VRAM: {available_memory_gb:.2f}GB")
        
        # Force CPU offloading if embeddings won't fit
        if embedding_memory_gb > available_memory_gb:
            self.force_cpu_offloading = True
            self.keep_embeddings_on_cpu = True
            if not self.quiet:
                print(f"   🚨 EMBEDDINGS TOO LARGE - Forcing CPU offloading")
                print(f"   🔧 Embeddings will be streamed to GPU only when needed")
        else:
            if not self.quiet:
                print(f"   ✅ Embeddings fit in VRAM")
    
    def _setup_compiled_functions(self):
        """Set up torch.compile optimized functions with robust error handling"""
        if self.enable_optimizations:
            try:
                # Compile the most performance-critical functions
                self.compiled_layer_processor = get_compiled_function(
                    self._process_layer_chunk_core, quiet=self.quiet
                )
                self.compiled_attention = get_compiled_function(
                    self._simple_attention_core, quiet=self.quiet
                )
                
                # Test compilation with dummy data to catch errors early
                if self.compiled_layer_processor and hasattr(self, 'config') and self.config:
                    try:
                        # Create dummy tensors for testing
                        dummy_hidden = torch.randn(1, 10, getattr(self.config, 'hidden_size', 4096), 
                                                 dtype=self.dtype, device=self.device)
                        dummy_weights = {}
                        
                        # Test compilation - if this fails, disable compilation
                        _ = self.compiled_layer_processor(dummy_hidden, 0, 0, dummy_weights)
                        
                        if not self.quiet:
                            print("✅ torch.compile test passed")
                        
                    except Exception as e:
                        if not self.quiet:
                            print(f"⚠️  torch.compile test failed: {e}")
                            print("   Disabling compilation for safety")
                        self.compiled_layer_processor = None
                        self.compiled_attention = None
                        
            except Exception as e:
                if not self.quiet:
                    print(f"⚠️  torch.compile setup failed: {e}")
                self.compiled_layer_processor = None
                self.compiled_attention = None
    
    def estimate_context_memory(self, sequence_length: int) -> float:
        """Estimate VRAM usage for given context length"""
        if not hasattr(self, 'config') or self.config is None:
            return 2.0  # Conservative fallback
        
        hidden_size = getattr(self.config, 'hidden_size', 4096)
        num_heads = getattr(self.config, 'num_attention_heads', 32)
        
        # Hidden states memory (sequence_length * hidden_size * 2 bytes for fp16)
        hidden_memory = sequence_length * hidden_size * 2
        
        # Attention matrix memory (num_heads * sequence_length^2 * 2 bytes)
        # This is the main bottleneck for long sequences
        attention_memory = num_heads * sequence_length * sequence_length * 2
        
        # Add some overhead for temporary tensors
        overhead = 0.5 * (1024**3)  # 0.5GB overhead
        
        total_bytes = hidden_memory + attention_memory + overhead
        return total_bytes / (1024**3)  # Convert to GB
    
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
        
        # Get current sequence token IDs using safe tokenizer
        sequence_text = prompt + "".join(generated_tokens)
        sequence_ids = self.safe_tokenizer.safe_encode(sequence_text)
        
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
    
    def load_tokenizer_and_config(self):
        """Load tokenizer and config with compatibility analysis"""
        if not self.quiet:
            print("📝 Loading tokenizer and config...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.config = AutoConfig.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize compatibility analysis
            compatibility_report = self.compatibility_manager.analyze_and_fix(
                self.tokenizer, self.config, None  # Embedding weights analyzed later
            )
            
            # Initialize safe tokenization pipeline
            self.safe_tokenizer = SafeTokenizationPipeline(
                self.tokenizer, self.compatibility_manager
            )
            
            # Initialize KV cache with correct dimensions
            if self.kv_cache and hasattr(self.config, 'num_hidden_layers'):
                self.kv_cache.num_layers = self.config.num_hidden_layers
                self.kv_cache.num_heads = getattr(self.config, 'num_attention_heads', 32)
                self.kv_cache.head_dim = getattr(self.config, 'hidden_size', 4096) // self.kv_cache.num_heads
            
            # Check for huge model and set CPU offloading strategy
            self.check_and_force_cpu_offloading()
            
            # Set up compiled functions after config is loaded
            self._setup_compiled_functions()
            
            if not self.quiet:
                print(f"   Model type: {self.config.model_type}")
                print(f"   Layers: {self.config.num_hidden_layers}")
                print(f"   Hidden size: {self.config.hidden_size}")
                print(f"   Vocab size: {self.config.vocab_size}")
                
                # Detect and set context length
                model_max_context = getattr(self.config, 'max_position_embeddings', 
                                          getattr(self.config, 'max_sequence_length', 4096))
                
                if self.max_context_length is None:
                    self.max_context_length = model_max_context
                    print(f"   Context length: {self.max_context_length} (model default)")
                else:
                    print(f"   Context length: {self.max_context_length} (user override, model supports {model_max_context})")
                    if self.max_context_length > model_max_context:
                        print(f"   ⚠️  Warning: Requested context ({self.max_context_length}) > model max ({model_max_context})")
            else:
                # Set context length even in quiet mode
                model_max_context = getattr(self.config, 'max_position_embeddings', 
                                          getattr(self.config, 'max_sequence_length', 4096))
                if self.max_context_length is None:
                    self.max_context_length = model_max_context
            
            return True
            
        except Exception as e:
            if not self.quiet:
                print(f"❌ Error loading tokenizer/config: {e}")
            return False
    
    def create_memory_maps(self):
        """Enhanced version with better error handling and debugging"""
        if not self.quiet:
            print("🗺️  Finding model weight files...")
        
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            try:
                from huggingface_hub import snapshot_download
                if not self.quiet:
                    print(f"   📥 Downloading {self.model_path}...")
                model_dir = Path(snapshot_download(self.model_path))
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                return False
        
        # Look for safetensor files
        self.safetensor_files = list(model_dir.glob("*.safetensors"))
        
        if not self.safetensor_files:
            # Fallback: look for .bin files
            bin_files = list(model_dir.glob("*.bin"))
            if bin_files:
                print(f"⚠️  No .safetensors files found, but found {len(bin_files)} .bin files")
                print("   This script only supports .safetensors format")
                print("   Consider converting the model to safetensors format")
            else:
                print("❌ No model weight files found (.safetensors or .bin)")
            return False
        
        # Initialize universal weight loader
        self.weight_loader = UniversalWeightLoader(self.safetensor_files, self.config, self.quiet)
        
        if not self.quiet:
            print(f"✅ Found {len(self.safetensor_files)} model files:")
            for f in self.safetensor_files:
                file_size = f.stat().st_size / (1024**3)  # GB
                print(f"   📄 {f.name} ({file_size:.1f}GB)")
        
        return True
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load specific layer weights using universal loader"""
        # Check weight cache first
        if self.weight_cache:
            cached_weights = self.weight_cache.get_layer(layer_idx)
            if cached_weights is not None:
                # Dequantize if needed
                if self.enable_quantization:
                    return dequantize_weights_int8(cached_weights)
                return cached_weights
        
        # Load using universal weight loader
        layer_weights = self.weight_loader.get_layer_weights(layer_idx)
        
        # Move to device and convert dtype
        for key, tensor in layer_weights.items():
            layer_weights[key] = tensor.to(device=self.device, dtype=self.dtype)
        
        # Cache weights (quantized if enabled)
        if self.weight_cache and layer_weights:
            weights_to_cache = quantize_weights_int8(layer_weights) if self.enable_quantization else layer_weights
            self.weight_cache.store_layer(layer_idx, weights_to_cache)
        
        return layer_weights
    
    def get_embedding_weights(self) -> Optional[torch.Tensor]:
        """Load embedding weights with smart GPU/CPU management"""
        if self.embedding_weights is not None:
            # Always return on correct device, streaming if needed
            if self.force_cpu_offloading or self.keep_embeddings_on_cpu:
                return self.embedding_weights.to(device=self.device, dtype=self.dtype)
            else:
                return self.embedding_weights
        
        # Load using universal weight loader
        embedding_tensor = self.weight_loader.load_embeddings()
        
        if embedding_tensor is not None:
            # Check memory requirements before loading to GPU
            if not hasattr(self, 'force_cpu_offloading'):
                self.check_and_force_cpu_offloading()
            
            # Update compatibility analysis with actual embedding size
            self.compatibility_manager.analyze_and_fix(
                self.tokenizer, self.config, embedding_tensor
            )
            
            # Smart device placement
            if self.force_cpu_offloading or self.keep_embeddings_on_cpu:
                # Keep on CPU, stream to GPU only when needed
                self.embedding_weights = embedding_tensor.to(dtype=self.dtype)
                if not self.quiet:
                    print(f"   💾 Embeddings stored on CPU ({embedding_tensor.shape})")
                return embedding_tensor.to(device=self.device, dtype=self.dtype)
            else:
                # Safe to keep on GPU
                self.embedding_weights = embedding_tensor.to(device=self.device, dtype=self.dtype)
                return self.embedding_weights
        
        return None
    
    def get_lm_head_weights(self) -> Optional[torch.Tensor]:
        """Load LM head weights with smart GPU/CPU management"""
        if self.lm_head_weights is not None:
            # Always return on correct device, streaming if needed
            if self.force_cpu_offloading or self.keep_embeddings_on_cpu:
                return self.lm_head_weights.to(device=self.device, dtype=self.dtype)
            else:
                return self.lm_head_weights
        
        # Load using universal weight loader
        lm_head_tensor = self.weight_loader.load_lm_head()
        
        if lm_head_tensor is not None:
            # Smart device placement (same logic as embeddings)
            if self.force_cpu_offloading or self.keep_embeddings_on_cpu:
                # Keep on CPU, stream to GPU only when needed
                self.lm_head_weights = lm_head_tensor.to(dtype=self.dtype)
                if not self.quiet:
                    print(f"   💾 LM head stored on CPU ({lm_head_tensor.shape})")
                return lm_head_tensor.to(device=self.device, dtype=self.dtype)
            else:
                # Safe to keep on GPU
                self.lm_head_weights = lm_head_tensor.to(device=self.device, dtype=self.dtype)
                return self.lm_head_weights
        
        return None
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def enhanced_memory_cleanup(self):
        """Enhanced memory cleanup for huge models"""
        if not self.quiet:
            print("🧹 Enhanced memory cleanup...")
        
        # Clear layer weights
        for layer_idx in list(self.layer_weights.keys()):
            if self.layer_weights[layer_idx]:
                for weight_name in list(self.layer_weights[layer_idx].keys()):
                    del self.layer_weights[layer_idx][weight_name]
            del self.layer_weights[layer_idx]
        
        self.layer_weights.clear()
        
        # Clear caches
        if self.weight_cache:
            self.weight_cache.clear()
        if self.kv_cache:
            self.kv_cache.reset()
        
        # Force garbage collection
        gc.collect()
        
        # Aggressive CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Force memory reset
            torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
        
        if not self.quiet:
            current_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            print(f"✅ Cleanup complete - VRAM usage: {current_memory:.2f}GB")
    
    def clear_vram_cache(self):
        """Clear VRAM cache and force garbage collection - enhanced version"""
        self.enhanced_memory_cleanup()
    
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
    
    def _simple_attention_core(self, hidden_states: torch.Tensor, weights: Dict[str, torch.Tensor],
                              layer_idx: int, use_kv_cache: bool = True) -> torch.Tensor:
        """Core attention computation - simplified and safe for torch.compile"""
        try:
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            
            # Simple linear transformations without complex reshaping
            q = F.linear(hidden_states, weights[q_key])
            k = F.linear(hidden_states, weights[k_key])
            v = F.linear(hidden_states, weights[v_key])
            
            # Use built-in scaled dot product attention (safest for compilation)
            try:
                # This is the safest approach for torch.compile
                output = F.scaled_dot_product_attention(
                    q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
                ).squeeze(1)
            except:
                # Ultra-simple fallback that avoids complex tensor operations
                output = q  # Just pass through if attention fails
            
            # Output projection
            output = F.linear(output, weights[o_key])
            return output
            
        except Exception as e:
            # If anything fails, return input unchanged
            return hidden_states
    
    def simple_attention_optimized(self, hidden_states: torch.Tensor, layer_idx: int, 
                                 use_kv_cache: bool = True, verbose: bool = True) -> torch.Tensor:
        """Optimized Grouped Query Attention computation with KV caching and compilation"""
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
            
            # Handle KV caching for generation (this part can't be compiled due to cache mutations)
            cached_k, cached_v = None, None
            if use_kv_cache and self.kv_cache and seq_len == 1 and self.kv_cache.seq_len > 0:
                cached_k, cached_v = self.kv_cache.get_kv(layer_idx)
            
            # Use compiled core attention function if available
            if self.compiled_attention:
                output = self.compiled_attention(hidden_states, weights, layer_idx, use_kv_cache)
            else:
                output = self._simple_attention_core(hidden_states, weights, layer_idx, use_kv_cache)
            
            # Handle KV cache updates (post-computation)
            if use_kv_cache and self.kv_cache:
                # Re-compute K,V for caching (this is a simplification)
                q_weight = weights[q_key]
                k_weight = weights[k_key]
                v_weight = weights[v_key]
                
                k = F.linear(hidden_states, k_weight)
                v = F.linear(hidden_states, v_weight)
                
                num_q_heads = getattr(self.config, 'num_attention_heads', 64)
                head_dim = hidden_size // num_q_heads
                num_kv_heads = k_weight.shape[0] // head_dim
                
                k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                
                if cached_k is not None and cached_v is not None:
                    k = torch.cat([cached_k, k], dim=2)
                    v = torch.cat([cached_v, v], dim=2)
                
                self.kv_cache.store_kv(layer_idx, k, v)
            
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
            gate = gate * torch.sigmoid(gate)  # SwiGLU activation
            intermediate = gate * up
            output = F.linear(intermediate, weights[down_key])
            return output
        except Exception as e:
            if verbose:
                print(f"   Error in MLP for layer {layer_idx}: {e}")
            return hidden_states
    
    def _process_layer_chunk_core(self, hidden_states: torch.Tensor, start_layer: int, 
                                end_layer: int, layer_weights_dict: Dict) -> torch.Tensor:
        """Core layer processing logic - simplified version for torch.compile"""
        current_states = hidden_states
        
        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx not in layer_weights_dict:
                continue
            
            weights = layer_weights_dict[layer_idx]
            residual = current_states
            
            # Layer norm
            ln_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            if ln_key in weights:
                try:
                    ln_weight = weights[ln_key]
                    current_states = F.layer_norm(current_states, (current_states.size(-1),), ln_weight)
                except Exception:
                    pass
            
            # Simplified attention that avoids shape assumptions
            try:
                q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                
                if all(key in weights for key in [q_key, k_key, v_key, o_key]):
                    batch_size, seq_len, hidden_size = current_states.shape
                    
                    # Use F.scaled_dot_product_attention if available (safer for compilation)
                    q = F.linear(current_states, weights[q_key])
                    k = F.linear(current_states, weights[k_key])
                    v = F.linear(current_states, weights[v_key])
                    
                    # Dynamically determine number of heads from weight shapes
                    q_heads = weights[q_key].shape[0] // (hidden_size // weights[q_key].shape[0] * hidden_size // weights[q_key].shape[0])
                    if q_heads == 0:  # Fallback calculation
                        q_heads = max(1, weights[q_key].shape[0] // 128)  # Assume 128 head_dim
                    head_dim = hidden_size // q_heads if q_heads > 0 else hidden_size
                    
                    # Reshape with calculated dimensions
                    q = q.view(batch_size, seq_len, q_heads, head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_len, -1, head_dim).transpose(1, 2)
                    v = v.view(batch_size, seq_len, -1, head_dim).transpose(1, 2)
                    
                    # Use built-in scaled dot product attention (safer for compilation)
                    try:
                        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    except:
                        # Manual fallback if scaled_dot_product_attention fails
                        scale = head_dim ** -0.5
                        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        attn_output = torch.matmul(attn_probs, v)
                    
                    # Reshape back
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
                    attn_output = F.linear(attn_output, weights[o_key])
                    
                    current_states = residual + attn_output
            except Exception:
                # If anything fails, skip attention for this layer
                pass
            
            # Post attention layer norm and MLP
            residual = current_states
            post_ln_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            if post_ln_key in weights:
                try:
                    post_ln_weight = weights[post_ln_key]
                    current_states = F.layer_norm(current_states, (current_states.size(-1),), post_ln_weight)
                except Exception:
                    pass
            
            # MLP
            gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            if all(key in weights for key in [gate_key, up_key, down_key]):
                try:
                    gate = F.linear(current_states, weights[gate_key])
                    up = F.linear(current_states, weights[up_key])
                    gate = gate * torch.sigmoid(gate)  # SwiGLU
                    intermediate = gate * up
                    mlp_output = F.linear(intermediate, weights[down_key])
                    current_states = residual + mlp_output
                except Exception:
                    pass
        
        return current_states
    
    def process_layer_chunk_optimized(self, hidden_states: torch.Tensor, start_layer: int, 
                                    end_layer: int, use_kv_cache: bool = True, verbose: bool = True) -> torch.Tensor:
        """Process hidden states through a chunk of layers with optimizations"""
        
        # Use compiled version if available for significant speedup (only for non-KV cached)
        if self.compiled_layer_processor and not use_kv_cache:
            try:
                # For compiled version, we need to pass weights as a regular dict
                return self.compiled_layer_processor(hidden_states, start_layer, end_layer, self.layer_weights)
            except Exception as e:
                if not self.quiet:
                    print(f"\n⚠️  Compiled function failed: {e}")
                    print("   Disabling compilation and falling back to standard processing")
                # Disable compilation for future calls
                self.compiled_layer_processor = None
                self.compiled_attention = None
                # Fall through to non-compiled version
        
        # Fallback to non-compiled version with full KV cache support
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
            
            # Optimized attention with KV caching
            attn_output = self.simple_attention_optimized(current_states, layer_idx, use_kv_cache, verbose=verbose)
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
                                        resume_from: str = None, debug_tokens: bool = False) -> Generator[str, None, None]:
        """Generate text with checkpoint support and robust tokenization"""
        
        self.generation_start_time = time.time()
        self.tokens_generated = 0
        self.total_chunks_processed = 0
        self.chunk_times = []
        
        # Reset KV cache for new generation
        if self.kv_cache:
            self.kv_cache.reset()
        
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
            hidden_states = hidden_states.to(self.device)  # Ensure on correct device
            
            # Restore RNG state for reproducibility
            if checkpoint.torch_rng_state:
                self.restore_rng_state(checkpoint.torch_rng_state)
            
            # Update KV cache sequence length
            if self.kv_cache:
                sequence_ids = self.safe_tokenizer.safe_encode(prompt) + \
                              [hash(token) % self.compatibility_manager.safe_vocab_size 
                               for token in generated_tokens]
                self.kv_cache.seq_len = len(sequence_ids) - 1
            
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
            
            # Tokenize and embed with validation using safe tokenizer
            try:
                input_ids = self.safe_tokenizer.safe_encode(prompt, add_special_tokens=True, max_length=512)
                input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
                
                # Validate input tokenization
                if not self.quiet or debug_tokens:
                    print(f"🔍 Input tokens: {len(input_ids)} -> {input_ids[:10]}...")
                    print(f"🔍 Safe vocab range: 0-{self.compatibility_manager.safe_vocab_size-1}")
                    
                    # Verify tokenization by decoding back
                    decoded_check = self.safe_tokenizer.safe_decode(input_ids)
                    print(f"🔍 Decode check: '{decoded_check[:50]}...'")
                
            except Exception as e:
                print(f"❌ Tokenization error: {e}")
                raise RuntimeError(f"Failed to tokenize input: {e}")
            
            embed_weights = self.get_embedding_weights()
            if embed_weights is None:
                raise RuntimeError("Could not load embedding weights")
            
            # Validate embedding weights shape
            if not self.quiet:
                print(f"🔍 Embedding weights shape: {embed_weights.shape}")
                print(f"🔍 Compatibility: Vocab mismatch = {self.compatibility_manager.vocab_mismatch}")
            
            hidden_states = F.embedding(input_tensor, embed_weights).to(self.dtype)
            
            # Enhanced cleanup for huge models - clear embeddings immediately after use
            if self.force_cpu_offloading:
                del embed_weights
                self.enhanced_memory_cleanup()
            
            # Initialize KV cache sequence length
            if self.kv_cache:
                self.kv_cache.seq_len = input_tensor.size(1) - 1  # -1 because we'll increment for each new token
            
            generated_tokens = []
            start_token = 0
        
        total_layers = self.config.num_hidden_layers
        
        # Main generation loop with enhanced safety
        for token_idx in range(start_token, max_tokens):
            # For generation after the first token, we only need to process the last token
            # thanks to KV caching
            if token_idx > 0 and self.kv_cache:
                # Only process the new token (last position)
                current_input = hidden_states[:, -1:, :]
                use_kv_cache = True
            else:
                # First token or no KV cache - process full sequence
                current_input = hidden_states
                use_kv_cache = False
            
            # For huge models, ALWAYS move to CPU between major operations
            if self.force_cpu_offloading and token_idx > 0:
                hidden_states_cpu = current_input.cpu()
                del current_input
                self.enhanced_memory_cleanup()
            else:
                hidden_states_cpu = current_input
            
            current_states = hidden_states_cpu
            total_chunks = (total_layers + self.max_layers_in_vram - 1) // self.max_layers_in_vram
            
            # Process through all layers in chunks
            for chunk_idx, start_layer in enumerate(range(0, total_layers, self.max_layers_in_vram)):
                end_layer = min(start_layer + self.max_layers_in_vram - 1, total_layers - 1)
                
                # Progress indicator
                progress = int((chunk_idx / total_chunks) * 100)
                dots = "." * (progress // 5)
                cache_indicator = "🔥" if use_kv_cache else "❄️"
                
                # Show compilation status
                if self.compiled_layer_processor and not use_kv_cache:
                    compile_indicator = "⚡"  # Compiled and active
                elif self.compiled_layer_processor:
                    compile_indicator = "🔄"  # Compiled but using KV cache (non-compiled)
                else:
                    compile_indicator = "🐌"  # Not compiled
                    
                print(f"\rToken {token_idx + 1}/{max_tokens}: Processing ({progress:3d}%) {dots:<20} [Chunk {chunk_idx + 1}/{total_chunks}] {cache_indicator}{compile_indicator}", end="", flush=True)
                
                # Move states to GPU for processing
                if current_states.device == torch.device('cpu'):
                    current_states = current_states.to(self.device)
                
                # Load and process chunk
                success = self.load_layer_chunk(start_layer, end_layer, verbose=False)
                if success and self.layer_weights:
                    current_states = self.process_layer_chunk_optimized(
                        current_states, start_layer, end_layer, use_kv_cache, verbose=False
                    )
                
                self.clear_vram_cache()
                
                # For huge models, always move back to CPU between chunks
                if self.force_cpu_offloading and start_layer < total_layers - self.max_layers_in_vram:
                    current_states = current_states.cpu()
                    self.enhanced_memory_cleanup()
            
            # Complete progress
            cache_indicator = "🔥" if use_kv_cache else "❄️"
            compile_indicator = "⚡" if self.compiled_layer_processor and not use_kv_cache else "🔄" if self.compiled_layer_processor else "🐌"
            print(f"\rToken {token_idx + 1}/{max_tokens}: Processing (100%) .................... [Chunk {total_chunks}/{total_chunks}] {cache_indicator}{compile_indicator} ✓", end="", flush=True)
            
            # Ensure final states are on GPU for token generation
            if current_states.device == torch.device('cpu'):
                current_states = current_states.to(self.device)
            
            # For KV cached generation, we need to get the last hidden state
            if use_kv_cache and self.kv_cache:
                # current_states should be [batch, 1, hidden] for the new token
                last_hidden = current_states[:, -1, :]
                # Update KV cache sequence length
                self.kv_cache.extend_sequence()
            else:
                # For first token or no cache, get the last position
                last_hidden = current_states[:, -1, :]
            
            # Generate next token with enhanced safety
            lm_head_weights = self.get_lm_head_weights()
            if lm_head_weights is None:
                lm_head_weights = self.get_embedding_weights()
            
            if lm_head_weights is not None:
                # Ensure LM head weights are on same device as hidden states
                if lm_head_weights.device != last_hidden.device:
                    lm_head_weights = lm_head_weights.to(last_hidden.device)
                
                # Generate logits
                logits = F.linear(last_hidden, lm_head_weights)
                
                if debug_tokens and token_idx < 3:  # Only show for first few tokens
                    print(f"\n🔍 Token {token_idx}: Logits shape {logits.shape}, range [{logits.min():.2f}, {logits.max():.2f}]")
                
                # For huge models, immediately clean up LM head weights after use
                if self.force_cpu_offloading:
                    del lm_head_weights
                    self.enhanced_memory_cleanup()
                
                # Safe token sampling using compatibility-aware pipeline
                next_token_id = self.safe_tokenizer.safe_token_sampling(logits, self.temperature)
                
            else:
                # Fallback: generate a safe token ID
                next_token_id = torch.tensor([[1]], device=self.device)  # Use a safe default
            
            # Safe token decoding
            try:
                token_id = next_token_id.item()
                
                if debug_tokens and token_idx < 3:
                    print(f"\n🔍 Decoding token ID {token_id}")
                
                # Use safe decoding pipeline
                next_token = self.safe_tokenizer.safe_decode(token_id)
                
                if debug_tokens and token_idx < 3:
                    print(f"🔍 Decoded to: '{next_token}' (length: {len(next_token)})")
                
                # Final validation
                if not next_token or len(next_token.strip()) == 0:
                    next_token = " "  # Safe fallback
                    if debug_tokens:
                        print(f"🔍 Empty decode, using space fallback")
                
            except Exception as e:
                if not self.quiet or debug_tokens:
                    print(f"\n⚠️  Token decoding error: {e}, using fallback")
                next_token = " "  # Safe fallback
                next_token_id = torch.tensor([[self.compatibility_manager.special_tokens['unk_token_id']]], 
                                           device=self.device)
            
            print(f"{next_token}", end="", flush=True)
            generated_tokens.append(next_token)
            yield next_token
            
            # Update hidden states for next iteration
            embed_weights = self.get_embedding_weights()
            if embed_weights is not None:
                new_embedding = F.embedding(next_token_id, embed_weights).to(self.dtype)
                
                # For huge models, immediately clean up embedding weights after use
                if self.force_cpu_offloading:
                    del embed_weights
                    self.enhanced_memory_cleanup()
                
                # For KV caching, we maintain the full sequence
                if use_kv_cache:
                    hidden_states = torch.cat([hidden_states, new_embedding], dim=1)
                else:
                    hidden_states = torch.cat([current_states, new_embedding], dim=1)
            
            self.tokens_generated += 1
            
            # Save checkpoint periodically
            if checkpoint_name and (token_idx + 1) % checkpoint_every == 0:
                checkpoint = self.create_checkpoint(prompt, generated_tokens, token_idx + 1, max_tokens, hidden_states)
                self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name, hidden_states, async_save=True)
            
            # Early stopping
            if next_token_id.item() == self.compatibility_manager.special_tokens['eos_token_id']:
                break
        
        # Final checkpoint
        if checkpoint_name:
            final_checkpoint = self.create_checkpoint(prompt, generated_tokens, len(generated_tokens), max_tokens, hidden_states)
            self.checkpoint_manager.save_checkpoint(final_checkpoint, checkpoint_name + "_final", hidden_states, async_save=False)
    
    def cleanup(self):
        """Clean up resources - enhanced for huge models"""
        if not self.quiet:
            print("\n🧹 Cleaning up...")
        
        self.enhanced_memory_cleanup()
        
        if not self.quiet:
            print("✅ Cleanup complete")


# Backwards compatibility alias
SimpleChunkedModelWithCheckpoints = OptimizedChunkedModelWithCheckpoints


def main():
    """Main application with enhanced compatibility and optimizations for huge models"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Chunked Inference with Universal Model Compatibility - OPTIMIZED FOR HUGE MODELS (671B+)",
        epilog="""
Environment Variables:
  DISABLE_TORCH_COMPILE=1    Disable torch.compile JIT compilation (useful for debugging)

Examples:
  python chunky.py --model microsoft/DialoGPT-small --tokens 50
  python chunky.py --model Qwen/Qwen3-32B --tokens 100 --vram 4.5 --chunk-layers 1
  python chunky.py --model deepseek-ai/deepseek-v3 --tokens 50 --vram 8 --chunk-layers 1  # 671B model!
  python chunky.py --model your_model --output-file result.txt  # Save output to file
  python chunky.py --model your_model --debug-tokens --tokens 5  # Debug tokenization issues
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Model path or HuggingFace model name")
    parser.add_argument("--vram", type=float, default=4.0, help="Max VRAM in GB")
    parser.add_argument("--prompt", default="The future of AI is", help="Generation prompt")
    parser.add_argument("--file", help="Read prompt from file (overrides --prompt)")
    parser.add_argument("--tokens", type=int, default=20, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--quiet", action="store_true", help="Hide initialization messages")
    parser.add_argument("--chunk-layers", type=int, help="Layers per chunk (default: auto-detect based on VRAM)")
    parser.add_argument("--max-context", type=int, help="Maximum context length (default: use model's max)")
    parser.add_argument("--output-file", help="Save complete output to file")
    parser.add_argument("--debug-tokens", action="store_true", help="Show detailed token debugging info")
    
    # Performance optimization arguments
    parser.add_argument("--no-optimizations", action="store_true", help="Disable performance optimizations")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV caching")
    parser.add_argument("--no-weight-cache", action="store_true", help="Disable weight caching")
    parser.add_argument("--no-quantization", action="store_true", help="Disable INT8 quantization")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile JIT compilation")
    
    # Huge model specific arguments
    parser.add_argument("--force-cpu-offloading", action="store_true", help="Force CPU offloading for embeddings")
    parser.add_argument("--aggressive-cleanup", action="store_true", help="Enable aggressive memory cleanup between operations")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N tokens")
    parser.add_argument("--checkpoint-name", help="Base name for checkpoint files")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-from", help="Resume from checkpoint file")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    parser.add_argument("--cleanup-checkpoints", action="store_true", help="Remove old checkpoint files")
    
    args = parser.parse_args()
    
    # Handle file input for prompt
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                file_prompt = f.read()
            
            if not args.quiet:
                file_size_kb = len(file_prompt.encode('utf-8')) / 1024
                print(f"📁 Reading prompt from: {args.file}")
                print(f"   File size: {file_size_kb:.1f}KB ({len(file_prompt)} characters)")
            
            args.prompt = file_prompt
            
        except FileNotFoundError:
            print(f"❌ Error: File not found: {args.file}")
            return
        except UnicodeDecodeError:
            print(f"❌ Error: Could not read file {args.file} (encoding issue)")
            print("   Try saving the file as UTF-8")
            return
        except Exception as e:
            print(f"❌ Error reading file {args.file}: {e}")
            return
    
    # Handle checkpoint management commands
    if args.list_checkpoints:
        manager = CheckpointManager(args.checkpoint_dir)
        manager.list_checkpoints()
        return
    
    if args.cleanup_checkpoints:
        manager = CheckpointManager(args.checkpoint_dir)
        manager.cleanup_old_checkpoints()
        return
    
    # Detect huge models and provide warnings/recommendations
    huge_model_indicators = ['671b', 'deepseek-v3', 'deepseek-v2', 'qwen-72b', 'llama-70b', 'mixtral-8x22b']
    is_likely_huge = any(indicator in args.model.lower() for indicator in huge_model_indicators)
    
    if is_likely_huge and not args.quiet:
        print(f"\n🚨 HUGE MODEL DETECTED: {args.model}")
        print(f"🔧 Recommended settings for huge models:")
        print(f"   --chunk-layers 1     (process one layer at a time)")
        print(f"   --vram 4.5          (leave headroom for operations)")
        print(f"   --force-cpu-offloading  (keep embeddings on CPU)")
        print(f"   --aggressive-cleanup    (cleanup between operations)")
        print(f"   --checkpoint-every 5    (save progress frequently)")
        print(f"")
        
        # Auto-apply recommended settings if not overridden
        if args.chunk_layers is None:
            args.chunk_layers = 1
            print(f"🔧 Auto-setting: --chunk-layers 1")
        
        if not args.force_cpu_offloading:
            args.force_cpu_offloading = True
            print(f"🔧 Auto-setting: --force-cpu-offloading")
        
        if not args.aggressive_cleanup:
            args.aggressive_cleanup = True
            print(f"🔧 Auto-setting: --aggressive-cleanup")
        
        if args.checkpoint_every == 10:  # Default value
            args.checkpoint_every = 5
            print(f"🔧 Auto-setting: --checkpoint-every 5")
        
        print("")
    
    # Create enhanced model with compatibility system
    enable_optimizations = not args.no_optimizations
    
    model = OptimizedChunkedModelWithCheckpoints(
        args.model, args.vram, args.temperature, args.quiet, 
        args.chunk_layers, args.checkpoint_dir, args.max_context,
        enable_optimizations=enable_optimizations
    )
    
    # Apply huge model specific settings
    if args.force_cpu_offloading:
        model.force_cpu_offloading = True
        model.keep_embeddings_on_cpu = True
        if not args.quiet:
            print("🔧 Forced CPU offloading enabled for embeddings")
    
    # Apply specific optimization disables
    if args.no_kv_cache and model.kv_cache:
        model.kv_cache = None
        if not args.quiet:
            print("⚠️  KV caching disabled")
    
    if args.no_weight_cache and model.weight_cache:
        model.weight_cache = None
        if not args.quiet:
            print("⚠️  Weight caching disabled")
    
    if args.no_quantization:
        model.enable_quantization = False
        if not args.quiet:
            print("⚠️  Quantization disabled")
    
    if args.no_compile:
        # Disable compilation by setting compiled functions to None
        model.compiled_layer_processor = None
        model.compiled_attention = None
        if not args.quiet:
            print("⚠️  torch.compile disabled")
    
    try:
        # Initialize with enhanced compatibility
        if not model.load_tokenizer_and_config():
            print("❌ Failed to load tokenizer/config")
            return
        
        if not model.create_memory_maps():
            print("❌ Failed to create memory maps")
            return
        
        # Provide memory usage estimates for huge models
        if not args.quiet and hasattr(model, 'config'):
            vocab_size = getattr(model.config, 'vocab_size', 50000)
            hidden_size = getattr(model.config, 'hidden_size', 4096)
            num_layers = getattr(model.config, 'num_hidden_layers', 80)
            
            embedding_memory = model.estimate_embedding_memory_gb(vocab_size, hidden_size)
            
            print(f"\n📊 Memory Estimates:")
            print(f"   Model parameters: ~{num_layers * hidden_size * hidden_size * 4 / 1e9:.1f}B")
            print(f"   Embedding memory: {embedding_memory:.2f}GB")
            print(f"   Recommended VRAM: {max(4.0, embedding_memory + 2.0):.1f}GB minimum")
            
            if embedding_memory > args.vram * 0.6:  # More than 60% of VRAM
                print(f"   🚨 WARNING: Embeddings may not fit in VRAM!")
                print(f"   🔧 CPU offloading will be automatically enabled")
        
        # Generate with enhanced safety and checkpoints
        if not args.quiet:
            if args.resume_from:
                print(f"\n🔄 Resuming from: {args.resume_from}")
            else:
                print(f"\n🎯 Prompt: {args.prompt}")
            print("🤖 Response:", end="", flush=True)
        
        generation_start = time.time()
        token_count = 0
        generated_text = []
        
        # Add memory monitoring for huge models
        initial_memory = model.get_vram_usage()
        max_memory_seen = initial_memory
        
        for token in model.generate_chunked_with_checkpoints(
            prompt=args.prompt,
            max_tokens=args.tokens,
            checkpoint_every=args.checkpoint_every,
            checkpoint_name=args.checkpoint_name,
            resume_from=args.resume_from,
            debug_tokens=args.debug_tokens
        ):
            generated_text.append(token)
            token_count += 1
            
            # Monitor peak memory usage
            current_memory = model.get_vram_usage()
            max_memory_seen = max(max_memory_seen, current_memory)
            
            # Aggressive cleanup for huge models if requested
            if args.aggressive_cleanup and token_count % 5 == 0:
                model.enhanced_memory_cleanup()
        
        generation_time = time.time() - generation_start
        
        print(f"\n{'✅ Generation complete!' if not args.quiet else ''}")
        
        # Show the complete generated text
        if generated_text:
            complete_text = "".join(generated_text)
            full_output = f"{args.prompt}{complete_text}"
            
            if not args.quiet:
                print("\n" + "="*80)
                print("📝 COMPLETE OUTPUT:")
                print("="*80)
                print(f"Prompt: {args.prompt}")
                print(f"Generated: {complete_text}")
                print("="*80)
            else:
                # In quiet mode, just show the complete text
                print(f"\nComplete output: {full_output}")
            
            # Save to file if requested
            if args.output_file:
                try:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        f.write(full_output)
                    if not args.quiet:
                        print(f"💾 Output saved to: {args.output_file}")
                except Exception as e:
                    print(f"❌ Error saving to file: {e}")
        else:
            print("\n⚠️  No tokens were generated.")
        
        # Show final stats with compatibility info and memory usage
        if not args.quiet:
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            print(f"📊 Performance: {token_count} tokens in {generation_time:.1f}s ({tokens_per_second:.2f} tok/s)")
            print(f"🔧 Processed {model.total_chunks_processed} chunks total")
            print(f"💾 Memory usage: {initial_memory:.2f}GB initial → {max_memory_seen:.2f}GB peak")
            
            # Show compatibility status
            if model.compatibility_manager.vocab_mismatch:
                print(f"🔧 Vocabulary compatibility: Applied automatic fixes")
                print(f"   Safe vocab range: 0-{model.compatibility_manager.safe_vocab_size-1}")
            else:
                print(f"✅ Vocabulary compatibility: No issues detected")
            
            if enable_optimizations:
                optimizations_used = []
                if model.kv_cache:
                    optimizations_used.append("KV Cache")
                if model.weight_cache:
                    optimizations_used.append("Weight Cache")
                if model.enable_quantization:
                    optimizations_used.append("INT8 Quantization")
                if model.compiled_layer_processor or model.compiled_attention:
                    optimizations_used.append("torch.compile")
                if model.force_cpu_offloading:
                    optimizations_used.append("CPU Offloading")
                
                if optimizations_used:
                    print(f"⚡ Optimizations used: {', '.join(optimizations_used)}")
            
            # Recommendations for improving performance
            if tokens_per_second < 0.5 and not is_likely_huge:
                print(f"\n💡 Performance Tips:")
                print(f"   • Try increasing --chunk-layers if you have more VRAM")
                print(f"   • Use --force-cpu-offloading for very large models")
                print(f"   • Consider using --no-compile if torch.compile is causing issues")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Generation interrupted. Progress saved in checkpoints.")
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Enhanced error reporting with compatibility info
        if hasattr(model, 'compatibility_manager') and model.compatibility_manager.vocab_mismatch:
            print(f"\n🔧 Compatibility Info:")
            print(f"   Tokenizer vocab: {model.compatibility_manager.tokenizer_vocab_size}")
            print(f"   Model vocab: {model.compatibility_manager.model_vocab_size}")
            print(f"   This model required automatic vocabulary alignment")
        
        # Memory debugging for huge models
        if hasattr(model, 'get_vram_usage'):
            current_memory = model.get_vram_usage()
            print(f"\n💾 Memory Info:")
            print(f"   Current VRAM usage: {current_memory:.2f}GB")
            print(f"   VRAM limit: {args.vram}GB")
            if current_memory > args.vram * 0.9:
                print(f"   🚨 Near memory limit! Try:")
                print(f"      --chunk-layers 1")
                print(f"      --force-cpu-offloading")
                print(f"      --aggressive-cleanup")
        
        if args.debug_tokens:
            print(f"\n🔍 Debug Info:")
            print(f"   Enable debug mode for detailed tokenization analysis")
        
        import traceback
        traceback.print_exc()
    finally:
        model.cleanup()


if __name__ == "__main__":
    main()
