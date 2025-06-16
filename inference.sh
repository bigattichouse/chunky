clear
#python simple_chunked_inference.py   --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B   --vram 4.0  --chunk-layers 2   --prompt "Explain artificial intelligence in detail"   --tokens 100   --quiet
#Qwen/Qwen3-32B
python chunky.py \
  --model Qwen/Qwen3-32B \
  --vram 5 \
  --chunk-layers 2 \
  --prompt "Write a linux commandline tetris program in C." \
  --tokens 1000 \
  --checkpoint-every 25 \
  --max-context 1024 \
  --checkpoint-name quantum_essay \
  --quiet
