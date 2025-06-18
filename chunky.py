#!/usr/bin/env python3
"""
True Model Chunking with CPU Offloading - Run 70B+ models in limited VRAM
Uses system RAM for storage and async loading for better performance
"""

import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
from tqdm import tqdm
import psutil
import threading
from queue import Queue
import time


@dataclass
class ChunkInfo:
    """Information about a model chunk"""
    chunk_id: int
    start_layer: int
    end_layer: int
    param_count: int
    estimated_memory_mb: float


@dataclass
class SystemInfo:
    """System resource information"""
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    available_vram_gb: float
    cpu_cores: int


class CPUOffloadedTensor:
    """Wrapper for tensors that live on CPU but can be moved to GPU on demand"""
    def __init__(self, tensor: torch.Tensor, name: str):
        self.cpu_tensor = tensor.cpu().pin_memory()  # Pin for faster GPU transfer
        self.name = name
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        
    def to_gpu(self, device: torch.device) -> torch.Tensor:
        """Move to GPU (non-blocking if possible)"""
        return self.cpu_tensor.to(device, non_blocking=True)
    
    def slice_to_gpu(self, indices: Union[torch.Tensor, slice], device: torch.device) -> torch.Tensor:
        """Move only a slice to GPU to save memory"""
        # Ensure indices are on CPU for indexing
        if isinstance(indices, torch.Tensor) and indices.device != torch.device('cpu'):
            indices = indices.cpu()
        return self.cpu_tensor[indices].to(device, non_blocking=True)


class AsyncChunkLoader:
    """Asynchronously loads chunks while processing"""
    def __init__(self, model):
        self.model = model
        self.queue = Queue(maxsize=2)  # Buffer next 2 chunks
        self.loading_thread = None
        self.stop_loading = False
        
    def start_loading(self, chunk_sequence: List[ChunkInfo]):
        """Start async loading thread"""
        self.stop_loading = False
        self.loading_thread = threading.Thread(
            target=self._loading_worker,
            args=(chunk_sequence,)
        )
        self.loading_thread.start()
    
    def _loading_worker(self, chunk_sequence: List[ChunkInfo]):
        """Worker thread that loads chunks"""
        try:
            for chunk in chunk_sequence:
                if self.stop_loading:
                    break
                
                # Load chunk to CPU memory
                chunk_weights = {}
                for layer_idx in range(chunk.start_layer, chunk.end_layer + 1):
                    layer_weights = self.model._load_layer_weights_to_cpu(layer_idx)
                    chunk_weights[layer_idx] = layer_weights
                
                # Put in queue (blocks if queue is full)
                self.queue.put((chunk, chunk_weights))
        except Exception as e:
            print(f"❌ Async loader error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_next_chunk(self, timeout: float = 30.0) -> Tuple[ChunkInfo, Dict]:
        """Get next pre-loaded chunk with timeout"""
        try:
            return self.queue.get(timeout=timeout)
        except:
            raise RuntimeError("Async loader timeout - chunk loading took too long")
    
    def stop(self):
        """Stop loading thread"""
        self.stop_loading = True
        if self.loading_thread:
            self.loading_thread.join()


class LayerChunkedModelWithOffloading:
    """
    Enhanced chunked model with CPU offloading and async loading
    Can run 70B+ models with just 4-6GB VRAM
    """
    
    def __init__(self, model_path: str, max_vram_mb: int = 4096, 
                 layers_per_chunk: int = 1, use_cpu_offload: bool = True,
                 enable_async: bool = False, preload_to_cpu: bool = False):  # Disabled by default due to issues
        self.model_path = model_path
        self.max_vram_mb = max_vram_mb
        self.layers_per_chunk = layers_per_chunk
        self.use_cpu_offload = use_cpu_offload
        self.enable_async = enable_async
        self.preload_to_cpu = preload_to_cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.config = None
        self.tokenizer = None
        self.dtype = torch.float16
        
        # Offloaded components (stay on CPU)
        self.embedding_weights_cpu = None
        self.lm_head_weights_cpu = None
        self.final_norm_weights = None  # Small enough to keep on GPU
        
        # Chunking info
        self.chunks = []
        self.total_layers = 0
        
        # File handles
        self.safetensor_files = []
        self.file_handles = {}
        
        # System info
        self.system_info = self._get_system_info()
        
        # Simple layer cache to avoid reloading from disk
        self.layer_cache = {}  # layer_idx -> CPUOffloadedTensor dict
        
        # Async loader
        self.async_loader = AsyncChunkLoader(self) if enable_async else None
        
    def _get_system_info(self) -> SystemInfo:
        """Get system resource information"""
        ram = psutil.virtual_memory()
        
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_available = (torch.cuda.get_device_properties(0).total_memory - 
                            torch.cuda.memory_allocated()) / (1024**3)
        else:
            vram_total = 0
            vram_available = 0
        
        return SystemInfo(
            total_ram_gb=ram.total / (1024**3),
            available_ram_gb=ram.available / (1024**3),
            total_vram_gb=vram_total,
            available_vram_gb=vram_available,
            cpu_cores=psutil.cpu_count()
        )
    
    def initialize(self):
        """Initialize model with resource analysis"""
        print(f"🚀 Initializing chunked model with CPU offloading")
        print(f"   Model: {self.model_path}")
        
        # Show system resources
        print(f"\n💾 System Resources:")
        print(f"   RAM: {self.system_info.available_ram_gb:.1f}/{self.system_info.total_ram_gb:.1f}GB available")
        print(f"   VRAM: {self.system_info.available_vram_gb:.1f}/{self.system_info.total_vram_gb:.1f}GB available")
        print(f"   CPU cores: {self.system_info.cpu_cores}")
        print(f"   CPU offloading: {'Enabled' if self.use_cpu_offload else 'Disabled'}")
        print(f"   Async loading: {'Enabled (experimental)' if self.enable_async else 'Disabled'}")
        
        # Load config and tokenizer
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.total_layers = self.config.num_hidden_layers
        
        # Check for grouped query attention
        num_q_heads = getattr(self.config, 'num_attention_heads', 32)
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_q_heads)
        if num_kv_heads != num_q_heads:
            print(f"   Using Grouped Query Attention: {num_q_heads} Q heads, {num_kv_heads} KV heads")
        
        # Find model files
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            from huggingface_hub import snapshot_download
            print("   Downloading model files...")
            model_dir = Path(snapshot_download(self.model_path))
        
        self.safetensor_files = list(model_dir.glob("*.safetensors"))
        print(f"   Found {len(self.safetensor_files)} model files")
        
        # Open file handles
        for file_path in self.safetensor_files:
            self.file_handles[str(file_path)] = safe_open(file_path, framework="pt", device="cpu")
        
        # Plan chunks with better memory estimates
        self._plan_chunks_with_memory_check()
        
        # Load persistent components to CPU
        self._load_persistent_components_cpu()
        
        # Optionally preload all layers to CPU
        if self.preload_to_cpu and self.system_info.available_ram_gb > 20:
            print(f"\n💾 Preloading all layers to CPU RAM (this may take a minute)...")
            self._preload_all_layers()
        elif self.preload_to_cpu:
            print(f"   ⚠️  Not enough RAM for preloading (need >20GB free, have {self.system_info.available_ram_gb:.1f}GB)")
        
        print(f"✅ Initialized with {len(self.chunks)} chunks")
        
    def _plan_chunks_with_memory_check(self):
        """Plan chunks considering actual memory constraints"""
        self.chunks = []
        
        # Estimate memory needs
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        
        # Embedding and LM head memory (will be on CPU)
        embedding_memory_mb = (vocab_size * hidden_size * 2) / (1024 * 1024)
        
        print(f"\n📊 Memory Planning:")
        print(f"   Embedding size: {embedding_memory_mb:.1f}MB (will be on CPU)")
        print(f"   Per-layer estimate: {self._estimate_layer_params() * 2 / (1024 * 1024):.1f}MB")
        
        # Calculate safe VRAM allocation (leave some headroom)
        safe_vram_mb = self.max_vram_mb * 0.8  # Use only 80% to be safe
        activation_memory_mb = 500  # Reserve for activations and operations
        available_for_weights = safe_vram_mb - activation_memory_mb
        
        # Adjust layers per chunk based on available VRAM
        layer_memory_mb = self._estimate_layer_params() * 2 / (1024 * 1024)
        max_layers_in_vram = int(available_for_weights / layer_memory_mb)
        
        if max_layers_in_vram < self.layers_per_chunk:
            print(f"   ⚠️  Reducing layers per chunk from {self.layers_per_chunk} to {max_layers_in_vram}")
            print(f"      Each layer needs ~{layer_memory_mb:.0f}MB, you have ~{available_for_weights:.0f}MB available")
            self.layers_per_chunk = max(1, max_layers_in_vram)
        
        # Create chunks
        chunk_id = 0
        for start_layer in range(0, self.total_layers, self.layers_per_chunk):
            end_layer = min(start_layer + self.layers_per_chunk - 1, self.total_layers - 1)
            
            params_per_layer = self._estimate_layer_params()
            chunk_params = params_per_layer * (end_layer - start_layer + 1)
            memory_mb = (chunk_params * 2) / (1024 * 1024)
            
            chunk = ChunkInfo(
                chunk_id=chunk_id,
                start_layer=start_layer,
                end_layer=end_layer,
                param_count=chunk_params,
                estimated_memory_mb=memory_mb
            )
            self.chunks.append(chunk)
            chunk_id += 1
            
            if chunk_id <= 3 or chunk_id >= len(range(0, self.total_layers, self.layers_per_chunk)) - 1:
                print(f"   Chunk {chunk.chunk_id}: layers {start_layer}-{end_layer}, ~{memory_mb:.1f}MB")
        
        if len(self.chunks) > 6:
            print(f"   ... ({len(self.chunks) - 6} more chunks)")
    
    def _estimate_layer_params(self) -> int:
        """Estimate parameters per transformer layer"""
        hidden_size = self.config.hidden_size
        intermediate_size = getattr(self.config, 'intermediate_size', hidden_size * 4)
        num_q_heads = getattr(self.config, 'num_attention_heads', 32)
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_q_heads)
        
        # For grouped query attention, k/v projections are smaller
        q_params = hidden_size * hidden_size
        kv_params = hidden_size * (hidden_size // num_q_heads * num_kv_heads) * 2
        o_params = hidden_size * hidden_size
        
        attn_params = q_params + kv_params + o_params
        mlp_params = hidden_size * intermediate_size * 3
        norm_params = hidden_size * 2
        
        return attn_params + mlp_params + norm_params
    
    def _load_persistent_components_cpu(self):
        """Load components to CPU memory with offloading wrapper"""
        print("\n📦 Loading persistent components to CPU...")
        
        # Find and load embeddings
        embed_patterns = ["model.embed_tokens.weight", "transformer.wte.weight", "embeddings.weight"]
        for pattern in embed_patterns:
            weight = self._find_and_load_weight(pattern)
            if weight is not None:
                self.embedding_weights_cpu = CPUOffloadedTensor(weight, "embeddings")
                print(f"   ✅ Embeddings loaded to CPU: {weight.shape}")
                del weight  # Free the original tensor
                break
        
        # Find and load LM head
        lm_head_patterns = ["lm_head.weight", "model.lm_head.weight", "output.weight"]
        for pattern in lm_head_patterns:
            weight = self._find_and_load_weight(pattern)
            if weight is not None:
                self.lm_head_weights_cpu = CPUOffloadedTensor(weight, "lm_head")
                print(f"   ✅ LM head loaded to CPU: {weight.shape}")
                del weight
                break
        
        # Final layer norm is small, can stay on GPU
        norm_patterns = ["model.norm.weight", "transformer.ln_f.weight", "final_layernorm.weight"]
        for pattern in norm_patterns:
            weight = self._find_and_load_weight(pattern)
            if weight is not None:
                self.final_norm_weights = weight.to(self.device, dtype=self.dtype)
                print(f"   ✅ Final norm loaded to GPU: {weight.shape}")
                break
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _preload_all_layers(self):
        """Preload all layer weights to CPU memory"""
        for layer_idx in tqdm(range(self.total_layers), desc="Preloading layers"):
            _ = self._load_layer_weights_to_cpu(layer_idx)
        
        # Calculate total memory used
        total_params = sum(
            sum(tensor.cpu_tensor.numel() for tensor in layer_weights.values())
            for layer_weights in self.layer_cache.values()
        )
        memory_gb = (total_params * 2) / (1024**3)  # 2 bytes per fp16 param
        print(f"   ✅ Preloaded {self.total_layers} layers using {memory_gb:.1f}GB RAM")
    
    def _find_and_load_weight(self, weight_name: str) -> Optional[torch.Tensor]:
        """Find and load a specific weight from safetensor files"""
        for file_handle in self.file_handles.values():
            if weight_name in file_handle.keys():
                return file_handle.get_tensor(weight_name)
        return None
    
    def _load_layer_weights_to_cpu(self, layer_idx: int) -> Dict[str, CPUOffloadedTensor]:
        """Load layer weights to CPU memory with caching"""
        # Check cache first
        if layer_idx in self.layer_cache:
            return self.layer_cache[layer_idx]
        
        layer_weights = {}
        
        patterns = {
            'q_proj': [f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            'k_proj': [f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            'v_proj': [f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            'o_proj': [f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            'gate_proj': [f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            'up_proj': [f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            'down_proj': [f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            'input_layernorm': [f"model.layers.{layer_idx}.input_layernorm.weight"],
            'post_attention_layernorm': [f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
        }
        
        for key, possible_names in patterns.items():
            for name in possible_names:
                weight = self._find_and_load_weight(name)
                if weight is not None:
                    layer_weights[key] = CPUOffloadedTensor(weight, f"layer_{layer_idx}_{key}")
                    break
        
        # Cache for future use
        if layer_weights:  # Only cache if we found some weights
            self.layer_cache[layer_idx] = layer_weights
        
        return layer_weights
    
    def _load_layer_weights_to_gpu(self, layer_weights_cpu: Dict[str, CPUOffloadedTensor]) -> Dict[str, torch.Tensor]:
        """Move layer weights from CPU to GPU"""
        gpu_weights = {}
        for key, cpu_tensor in layer_weights_cpu.items():
            gpu_weights[key] = cpu_tensor.to_gpu(self.device).to(self.dtype)
        return gpu_weights
    
    def _process_transformer_layer(self, hidden_states: torch.Tensor, layer_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process hidden states through a single transformer layer"""
        residual = hidden_states
        
        # Input layer norm
        if 'input_layernorm' in layer_weights:
            hidden_states = F.layer_norm(hidden_states, (hidden_states.size(-1),), layer_weights['input_layernorm'])
        
        # Self attention
        if all(k in layer_weights for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            # Project to Q, K, V
            q = F.linear(hidden_states, layer_weights['q_proj'])
            k = F.linear(hidden_states, layer_weights['k_proj'])
            v = F.linear(hidden_states, layer_weights['v_proj'])
            
            batch_size, seq_len = hidden_states.shape[:2]
            
            # Handle grouped query attention
            q_dim = q.shape[-1]
            k_dim = k.shape[-1]
            v_dim = v.shape[-1]
            
            num_q_heads = getattr(self.config, 'num_attention_heads', 32)
            num_kv_heads = getattr(self.config, 'num_key_value_heads', num_q_heads)
            
            q_head_dim = q_dim // num_q_heads
            kv_head_dim = k_dim // num_kv_heads
            
            # Reshape Q, K, V
            q = q.view(batch_size, seq_len, num_q_heads, q_head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
            
            # Handle grouped query attention
            if num_kv_heads != num_q_heads:
                repeat_factor = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            
            # Compute attention
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, q_dim)
            
            # Output projection
            attn_output = F.linear(attn_output, layer_weights['o_proj'])
            
            # Add residual
            hidden_states = residual + attn_output
        
        # Post attention layer norm + MLP
        residual = hidden_states
        
        if 'post_attention_layernorm' in layer_weights:
            hidden_states = F.layer_norm(hidden_states, (hidden_states.size(-1),), layer_weights['post_attention_layernorm'])
        
        if all(k in layer_weights for k in ['gate_proj', 'up_proj', 'down_proj']):
            gate = F.linear(hidden_states, layer_weights['gate_proj'])
            up = F.linear(hidden_states, layer_weights['up_proj'])
            gate = F.silu(gate)
            intermediate = gate * up
            mlp_output = F.linear(intermediate, layer_weights['down_proj'])
            hidden_states = residual + mlp_output
        
        return hidden_states
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8, 
                 show_memory: bool = False, show_progress: bool = True) -> str:
        """Generate text with CPU offloading and optional async loading"""
        print(f"\n🎯 Generating with {len(self.chunks)} chunks...")
        print("🤖 Response: ", end='', flush=True)
        
        # Clear VRAM before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if show_memory:
                initial_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"📊 Initial VRAM usage: {initial_vram_mb:.1f}MB")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")  # Shape: [1, seq_len]
        generated_ids = input_ids.clone()
        generated_text_parts = []  # Accumulate generated text
        
        # Generate token by token
        with torch.no_grad():
            for token_idx in range(max_tokens):
                if show_memory:
                    print(f"\n🔍 === Token {token_idx+1} ===")
                    if token_idx == 0:
                        print(f"   Input sequence length: {generated_ids.shape[1]}")
                        if self.layer_cache:
                            print(f"   ✅ {len(self.layer_cache)} layers already cached in CPU memory")
                
                # Move input to GPU only when needed
                current_ids = generated_ids.to(self.device)  # Shape: [1, seq_len]
                
                # Get embeddings - handle large embedding matrix
                with torch.no_grad():
                    # Check if we can fit embeddings in VRAM
                    seq_len = current_ids.shape[1]
                    embedding_slice_size_mb = (seq_len * self.config.hidden_size * 2) / (1024 * 1024)
                    
                    if torch.cuda.is_available():
                        free_memory_mb = (torch.cuda.get_device_properties(0).total_memory - 
                                        torch.cuda.memory_allocated()) / (1024 * 1024)
                        
                        if free_memory_mb > embedding_slice_size_mb * 2:  # Good margin
                            # Method 1: Do lookup on GPU (faster)
                            embeddings_gpu = self.embedding_weights_cpu.to_gpu(self.device)
                            hidden_states = F.embedding(current_ids, embeddings_gpu).to(self.dtype)
                            del embeddings_gpu
                        else:
                            # Method 2: Do lookup on CPU then move result (more memory efficient)
                            current_ids_cpu = current_ids.cpu()
                            embeddings_cpu = F.embedding(current_ids_cpu, self.embedding_weights_cpu.cpu_tensor)
                            hidden_states = embeddings_cpu.to(self.device, dtype=self.dtype)
                            del embeddings_cpu, current_ids_cpu
                    else:
                        # CPU only
                        hidden_states = F.embedding(current_ids, self.embedding_weights_cpu.cpu_tensor).to(self.dtype)
                
                # Process through each chunk
                for chunk_idx, chunk in enumerate(self.chunks):
                    # Show progress
                    if show_progress:
                        print(f"\r🔄 Token {token_idx+1}/{max_tokens}: Chunk {chunk_idx+1}/{len(self.chunks)}", end='', flush=True)
                    
                    # Always load synchronously for now (async has issues)
                    chunk_weights_cpu = {}
                    for layer_idx in range(chunk.start_layer, chunk.end_layer + 1):
                        chunk_weights_cpu[layer_idx] = self._load_layer_weights_to_cpu(layer_idx)
                    
                    # Process layers in chunk
                    for layer_idx in range(chunk.start_layer, chunk.end_layer + 1):
                        if layer_idx in chunk_weights_cpu:
                            # Move layer to GPU
                            layer_weights_gpu = self._load_layer_weights_to_gpu(chunk_weights_cpu[layer_idx])
                            
                            # Process
                            hidden_states = self._process_transformer_layer(hidden_states, layer_weights_gpu)
                            
                            # Free GPU memory immediately
                            for weight in layer_weights_gpu.values():
                                del weight
                            del layer_weights_gpu
                    
                    # Free CPU chunk dict (not the cached data)
                    del chunk_weights_cpu
                    
                    # Aggressive cleanup every few chunks
                    if chunk_idx % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # After all chunks processed
                if show_memory and token_idx == 1:
                    print(f"\n🔍 Token 2 Debug: All chunks processed, computing logits...")
                    if show_memory and chunk_idx % 10 == 0:
                        current_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        print(f"\n📊 VRAM after chunk {chunk_idx}: {current_vram_mb:.1f}MB")
                
                # Final layer norm
                if self.final_norm_weights is not None:
                    hidden_states = F.layer_norm(
                        hidden_states, (hidden_states.size(-1),), self.final_norm_weights
                    )
                
                # Get logits - handle large LM head carefully
                last_hidden = hidden_states[:, -1, :]  # Get last position [batch, hidden]
                
                if show_memory and token_idx == 1:
                    print(f"🔍 Token 2 Debug: Computing logits, last_hidden shape: {last_hidden.shape}")
                
                # Check if we have enough VRAM for LM head
                lm_head_size_mb = (self.lm_head_weights_cpu.cpu_tensor.numel() * 2) / (1024 * 1024)
                if torch.cuda.is_available():
                    free_memory_mb = (torch.cuda.get_device_properties(0).total_memory - 
                                    torch.cuda.memory_allocated()) / (1024 * 1024)
                    
                    if show_memory and token_idx == 1:
                        print(f"   LM head size: {lm_head_size_mb:.1f}MB, free VRAM: {free_memory_mb:.1f}MB")
                    
                    if free_memory_mb > lm_head_size_mb * 1.2:  # 20% safety margin
                        # Enough VRAM - compute on GPU
                        lm_head_gpu = self.lm_head_weights_cpu.to_gpu(self.device).to(self.dtype)
                        logits = F.linear(last_hidden, lm_head_gpu)  # [batch, vocab]
                        del lm_head_gpu
                    else:
                        # Not enough VRAM - compute on CPU
                        if show_memory and token_idx == 1:
                            print(f"   Using CPU for logits computation...")
                        last_hidden_cpu = last_hidden.cpu()
                        logits_cpu = F.linear(last_hidden_cpu, self.lm_head_weights_cpu.cpu_tensor)
                        logits = logits_cpu.to(self.device)  # [batch, vocab]
                        del last_hidden_cpu, logits_cpu
                else:
                    # CPU only mode
                    logits = F.linear(last_hidden, self.lm_head_weights_cpu.cpu_tensor)  # [batch, vocab]
                
                if show_memory and token_idx == 1:
                    print(f"   ✅ Logits computed: {logits.shape}")
                
                # Sample
                logits = logits / temperature  # Should be [batch, vocab]
                
                # Debug shape issues
                if show_memory and token_idx == 0:
                    print(f"\n🔍 Debug shapes: hidden_states={hidden_states.shape}, last_hidden={last_hidden.shape}, logits={logits.shape}")
                
                probs = F.softmax(logits, dim=-1)  # [batch, vocab]
                next_token = torch.multinomial(probs[0], 1)  # [1]
                
                if show_memory and token_idx == 1:
                    print(f"🔍 Token 2 Debug: Sampled token ID: {next_token.item()}")
                
                # Decode and print token
                token_text = self.tokenizer.decode(next_token.item())
                generated_text_parts.append(token_text)
                
                # Clear progress if shown
                if show_progress:
                    print(f"\r{' ' * 60}\r", end='', flush=True)
                
                # Print the actual token
                print(token_text, end='', flush=True)
                
                # Append to sequence
                next_token_expanded = next_token.unsqueeze(0)  # [1, 1]
                generated_ids = torch.cat([generated_ids, next_token_expanded.cpu()], dim=1)
                
                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Clear GPU memory
                del hidden_states, logits, probs, next_token
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if show_memory and token_idx == 1:
                    print(f"\n🔍 Token 2 Debug: Token generation complete, looping to token 3...")
        
        # Clear final progress line
        if show_progress:
            print(f"\r{' ' * 60}\r", end='', flush=True)
        
        # Return full generated text
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return full_text
    
    def cleanup(self):
        """Clean up resources"""
        for handle in self.file_handles.values():
            handle.__exit__(None, None, None)
        self.file_handles.clear()
        
        self.embedding_weights_cpu = None
        self.lm_head_weights_cpu = None
        self.final_norm_weights = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="True Model Chunking with CPU Offloading - Run 70B+ models with minimal VRAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - runs Qwen3 32B with 1 layer per chunk
  python chunky_offload.py --model Qwen/Qwen3-32B --layers-per-chunk 1
  
  # Preload all layers to RAM for faster generation (needs ~60GB free RAM)
  python chunky_offload.py --model Qwen/Qwen3-32B --layers-per-chunk 1 --preload
  
  # Run with more layers per chunk (faster but uses more VRAM)
  python chunky_offload.py --model Qwen/Qwen3-32B --layers-per-chunk 3
  
  # Debug hanging issues
  python chunky_offload.py --model Qwen/Qwen3-32B --layers-per-chunk 1 --show-memory --no-progress
  
  # Run 70B model
  python chunky_offload.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --layers-per-chunk 1

This version:
- Keeps embeddings and LM head in CPU RAM
- Loads model layers on-demand from disk (slow) or preloads to RAM (fast)
- Processes one chunk at a time to stay within VRAM limits
        """
    )
    
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--prompt", default="Write a Python function to calculate fibonacci numbers:", 
                        help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-vram", type=int, default=4096, help="Max VRAM to use in MB")
    parser.add_argument("--layers-per-chunk", type=int, default=1, help="Transformer layers per chunk")
    parser.add_argument("--no-cpu-offload", action="store_true", help="Disable CPU offloading")
    parser.add_argument("--async", action="store_true", help="Enable async chunk loading (experimental)")
    parser.add_argument("--preload", action="store_true", help="Preload all layers to CPU RAM")
    parser.add_argument("--show-memory", action="store_true", help="Show memory usage during generation")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display during generation")
    
    args = parser.parse_args()
    
    # Create model with offloading
    model = LayerChunkedModelWithOffloading(
        model_path=args.model,
        max_vram_mb=args.max_vram,
        layers_per_chunk=args.layers_per_chunk,
        use_cpu_offload=not args.no_cpu_offload,
        enable_async=  True,#args.async,  # Now opt-in instead of opt-out
        preload_to_cpu=args.preload
    )
    
    try:
        # Initialize
        model.initialize()
        
        # Generate
        print(f"\n📝 Prompt: {args.prompt}")
        print("🤖 Response: ", end='', flush=True)
        
        response = model.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            show_memory=args.show_memory,
            show_progress=not args.no_progress
        )
        
        print(f"\n\n✅ Generation complete!")
        
        # Show final memory stats
        if torch.cuda.is_available():
            max_vram = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"📊 Peak VRAM usage: {max_vram:.2f}GB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.cleanup()
        print("🧹 Cleaned up resources")


if __name__ == "__main__":
    main()
