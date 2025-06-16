clear
#python simple_chunked_inference.py   --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B   --vram 4.0  --chunk-layers 2   --prompt "Explain artificial intelligence in detail"   --tokens 100   --quiet
python chunky.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --vram 3.5 \
  --chunk-layers 2 \
  --prompt "Write a detailed 1000-word essay about quantum computing and its applications" \
  --tokens 1000 \
  --checkpoint-every 25 \
  --checkpoint-name quantum_essay \
  --quiet
