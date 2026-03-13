export PYTHONPATH="/Megatron-LM:$PYTHONPATH"
source scripts/models/qwen3-30B-A3B.sh

# 转换 HuggingFace 模型到 PyTorch 分布式格式
torchrun --nproc-per-node 2 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-30B-A3B/ \     # hf 源模型
   --save /root/Qwen3-30B-A3B_torch_dist/     # 保存 dist 格式地址

# --nproc-per-node means world_size also is pp_size.