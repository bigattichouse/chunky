# Chunky.py - Large Model Inference on Limited VRAM

**Run massive language models (70B-671B parameters) on consumer GPUs with as little as 4GB VRAM through intelligent chunking and checkpoint management.**

This is a work in progress. I'm still testing, but I have it at least generating tokens with the core library.

> **⚠️ Testing Status**: Currently tested on Ubuntu with DeepSeek-R1 70B. Other platforms and models likely work but are untested. Performance numbers are estimates.

## 🚀 What is Chunky?

Chunky enables you to run enormous language models that normally require 140GB+ of VRAM on your consumer GPU by:

- **Loading model layers one chunk at a time** into VRAM
- **Processing sequences through each chunk sequentially**  
- **Saving/resuming long generations** with comprehensive checkpoints
- **Optimizing memory usage** through CPU/GPU shuttling

**Example**: Run DeepSeek-R1 70B (normally needs 140GB VRAM) on a 5GB Quadro P2200!

## 📋 Requirements

### Essential Dependencies
```bash
# Install minimal requirements (what chunky.py actually needs)
pip install torch>=2.0.0 transformers>=4.36.0 safetensors>=0.4.0 numpy>=1.24.0 huggingface_hub>=0.19.0

# Or install from requirements.txt
pip install -r requirements.txt
```

### Optional Dependencies  
```bash
# Better performance on some systems
pip install accelerate>=0.24.0

# System monitoring (for debugging)
pip install psutil>=5.9.0 nvidia-ml-py3>=7.352.0
```

### Hardware Requirements
- **Minimum**: 4GB VRAM + 16GB RAM *(estimated)*
- **Tested**: 4GB VRAM + 128GB RAM *(Ubuntu)*
- **Recommended**: 6GB+ VRAM + 32GB+ RAM  
- **Storage**: 150GB+ free space *(DeepSeek-R1 70B is ~140GB)*

### Installation Notes

**Why so few dependencies?** Chunky is designed to be lightweight and avoid dependency hell. Unlike complex ML frameworks, we only use what's absolutely necessary:

- **No quantization libraries** (bitsandbytes, optimum) - We use native PyTorch precision
- **No acceleration frameworks** (DeepSpeed, FairScale) - Simple chunking is more reliable  
- **No specialized tokenizers** - Standard transformers tokenizers work fine
- **No monitoring dependencies** - Built-in VRAM tracking is sufficient

This means **faster installation**, **fewer version conflicts**, and **better reliability**.

### Supported Platforms
- **Linux**: Full support ✅ **(tested on Ubuntu)**
- **Windows**: Likely works but **untested**
- **macOS**: Likely works in CPU mode but **untested**

### Installation Troubleshooting
```bash
# If PyTorch installation fails, try with specific CUDA version (untested)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# If you get "No module named 'transformers'" 
pip install --upgrade transformers

# For M1/M2 Macs - CPU mode only (untested)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

## ⚡ Quick Start

### Basic Usage
```bash
# Simple generation
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --prompt "Explain quantum computing" \
  --tokens 50

# Optimized for your VRAM
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --vram 4.0 \
  --chunk-layers 2 \
  --prompt "Write a story about AI" \
  --tokens 200 \
  --quiet
```

### Long Generation with Checkpoints
```bash
# Start a long generation with auto-save
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --vram 3.5 \
  --chunk-layers 2 \
  --prompt "Write a detailed 1000-word essay about the future of artificial intelligence" \
  --tokens 1000 \
  --checkpoint-every 25 \
  --checkpoint-name ai_essay \
  --quiet

# Resume if interrupted
python chunky.py \
  --resume-from checkpoints/ai_essay_t025_20241215_143022.ckpt \
  --quiet
```

## 🔧 Command Line Options

### Core Options
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | **Required** | HuggingFace model name or local path |
| `--prompt` | "The future of AI is" | Generation prompt |
| `--tokens` | 20 | Number of tokens to generate |
| `--temperature` | 0.7 | Sampling temperature (0.1-2.0) |

### Memory Management
| Option | Default | Description |
|--------|---------|-------------|
| `--vram` | 4.0 | Maximum VRAM usage in GB |
| `--chunk-layers` | Auto | Layers per chunk (1-8) |
| `--quiet` | False | Hide initialization messages |

### Checkpoint System
| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint-every` | 10 | Save checkpoint every N tokens |
| `--checkpoint-name` | None | Base name for checkpoint files |
| `--checkpoint-dir` | ./checkpoints | Directory for checkpoint files |
| `--resume-from` | None | Resume from checkpoint file |

### Checkpoint Management
| Option | Description |
|--------|-------------|
| `--list-checkpoints` | Show all available checkpoints |
| `--cleanup-checkpoints` | Remove old checkpoint files |

## 📊 Performance Optimization

### Finding Optimal Chunk Size

Your VRAM usage depends on the chunk size. Larger chunks = fewer total chunks = faster generation:

```bash
# Conservative (40 chunks per token)
--chunk-layers 2  # ~3.5GB VRAM usage

# Aggressive (27 chunks per token) 
--chunk-layers 3  # ~5.2GB VRAM usage

# Monitor VRAM usage to find your sweet spot
nvidia-smi -l 1
```

### Performance Expectations

| Setup | Chunks/Token | Speed (est.) | VRAM Usage (est.) |
|-------|--------------|--------------|-------------------|
| 1 layer/chunk | 80 | ~0.3 tok/s | ~1.8GB |
| 2 layers/chunk | 40 | ~0.6 tok/s | ~3.5GB |
| 3 layers/chunk | 27 | ~0.9 tok/s | ~5.2GB |

*Note: Performance numbers are estimates. Actual speed depends on your GPU, CPU, and disk speed.*

### Memory Optimization Tips

```bash
# Set PyTorch memory management for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use more aggressive chunk size if you have VRAM headroom
python chunky.py --vram 6.0 --chunk-layers 3 [other options]

# For very long sequences, increase checkpoint frequency
python chunky.py --checkpoint-every 10 [other options]

# Clear GPU cache if you encounter memory issues
python -c "import torch; torch.cuda.empty_cache()"
```

## 🎯 Usage Examples

### 1. Academic Writing
```bash
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --prompt "Write a comprehensive literature review on machine learning in healthcare" \
  --tokens 800 \
  --checkpoint-every 50 \
  --checkpoint-name healthcare_review \
  --temperature 0.6 \
  --quiet
```

### 2. Creative Writing
```bash
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --prompt "Write a science fiction short story about first contact with aliens" \
  --tokens 1200 \
  --checkpoint-every 100 \
  --checkpoint-name scifi_story \
  --temperature 0.8
```

### 3. Technical Documentation
```bash
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --prompt "Create a detailed API documentation for a REST service" \
  --tokens 600 \
  --checkpoint-every 30 \
  --checkpoint-name api_docs \
  --temperature 0.4
```

### 4. Resuming Work
```bash
# List available checkpoints
python chunky.py --list-checkpoints

# Resume specific checkpoint
python chunky.py --resume-from checkpoints/scifi_story_t100_20241215_143022.ckpt
```

## 📁 Checkpoint System

### How Checkpoints Work

Chunky automatically saves your progress including:
- **Generation state**: Current position, generated text
- **Neural network state**: Hidden representations 
- **Model configuration**: Temperature, chunk size, etc.
- **Random state**: For reproducible generation

### Checkpoint Files

```
checkpoints/
├── ai_essay_t025_20241215_143022.ckpt     # Checkpoint at token 25
├── ai_essay_t025_20241215_143022_hidden.pt # Hidden states (large file)
├── ai_essay_t050_20241215_143522.ckpt     # Checkpoint at token 50
├── ai_essay_t050_20241215_143522_hidden.pt
└── checkpoint_metadata.json               # Index of all checkpoints
```

### Managing Checkpoints

```bash
# View all checkpoints with details
python chunky.py --list-checkpoints

# Clean up old checkpoints (keeps last 5)  
python chunky.py --cleanup-checkpoints

# Manual cleanup
rm -rf checkpoints/old_project_*
```

## 🛠️ Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Reduce chunk size
python chunky.py --chunk-layers 1 [other options]

# Reduce VRAM limit
python chunky.py --vram 3.0 [other options]

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**"No .safetensors files found"**
```bash
# Model download failed - check internet connection and disk space
# Clear HuggingFace cache and retry
rm -rf ~/.cache/huggingface/hub/
python chunky.py --model [model_name] [options]
```

**"ImportError: No module named..."**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Or install individually
pip install torch>=2.0.0 transformers>=4.36.0 safetensors>=0.4.0
```

**Slow generation**
```bash
# Increase chunk size for better performance
python chunky.py --chunk-layers 3 [other options]

# Use fewer tokens for testing
python chunky.py --tokens 10 [other options]
```

**Checkpoint loading errors**
```bash
# List checkpoints to find valid ones
python chunky.py --list-checkpoints

# Start fresh if checkpoint is corrupted
python chunky.py --checkpoint-name new_session [other options]
```

### Performance Issues

**Generation too slow:**
- Increase `--chunk-layers` if you have VRAM headroom
- Use `--quiet` mode to reduce output overhead
- Close other GPU applications

**Running out of disk space:**
- Clean up old checkpoints regularly
- Use external storage for checkpoint directory
- Reduce checkpoint frequency for shorter generations

**Memory leaks:**
- Restart script for very long sessions (1000+ tokens)
- Monitor system RAM usage alongside VRAM

## 🔬 Advanced Usage

### Custom Model Paths
```bash
# Local model directory
python chunky.py --model /path/to/local/model [options]

# Different model variants
python chunky.py --model microsoft/DialoGPT-large [options]
python chunky.py --model EleutherAI/gpt-j-6b [options]
```

### Batch Processing *(untested)*
```bash
# Process multiple prompts
for prompt in "$(cat prompts.txt)"; do
  python chunky.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --prompt "$prompt" \
    --tokens 100 \
    --checkpoint-name "batch_$(date +%s)" \
    --quiet
done
```

### Integration with Scripts *(untested)*
```bash
# Save output to file
python chunky.py [options] > output.txt 2>&1

# Use in pipelines
python chunky.py [options] | tee generation.log

# Background processing
nohup python chunky.py [options] &
```

## 📈 Model Support

### Tested Models
- ✅ **DeepSeek-R1 Series** (70B tested on Ubuntu)

### Likely Compatible Models *(untested)*
- **Llama 2/3 Series** (7B, 13B, 70B)
- **Mistral Series** (7B, 8x7B) 
- **Qwen Series** (Various sizes)

### Model Requirements
- Must use **safetensors** format
- Compatible with **transformers** library
- Standard transformer architecture (attention + MLP layers)

## ⚠️ Important Notes

### First Run
- **Download time**: 70B models are ~140GB *(actual download time varies by connection)*
- **Disk space**: Ensure you have 200GB+ free space *(for model + checkpoints)*
- **Network**: Stable internet connection required for initial download

### Long Generations
- Use `--checkpoint-every` for generations >100 tokens *(recommended)*
- Monitor disk space (checkpoints can be 2-4GB each) *(estimated)*
- Consider external storage for checkpoint directory

### Quality vs Speed
- Higher `--chunk-layers` = faster but needs more VRAM
- Lower `--temperature` = more focused but less creative
- Shorter prompts = faster processing

## 🤝 Contributing

Found a bug or want to contribute? 

1. Test with smaller models first (DialoGPT-small)
2. Include full command and error output
3. Specify your GPU model and VRAM amount
4. Check existing issues for similar problems

## 📜 License

MIT License - Use freely for research and commercial applications.

---

**Made something cool with Chunky? Share it!** This tool enables running cutting-edge AI on consumer hardware - democratizing access to powerful language models. 🚀
