#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

ray stop --force
rm -rf /tmp/ray

set -ex
ulimit -n 65535
ulimit -u 65535

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
#export PYTHONPATH="/Megatron-LM/:/sglang/python:$PYTHONPATH"
#export PYTHONPATH="/Megatron-LM:/sglang/python:/Megatron-Bridge:/MindSpeed:$PYTHONPATH"
export PYTHONPATH="/Megatron-Bridge/src:/Megatron-LM:/sglang/python:/MindSpeed:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/slime/scripts/models/qwen3-30B-A3B.sh"

#CKPT_ARGS=(
#   --hf-checkpoint /data/Qwen3-30B-A3B
#   --ref-load /data/Qwen3-30B-A3B_torch_dist_8
#   --load /data/Qwen3-30B-A3B
#   --save /data/Qwen3-30B-A3B_slime_8
#   --save-interval 500
#)

#CKPT_ARGS=(
#   --hf-checkpoint /mnt/share/m00876805/ckpt/Qwen3-30B-MoE
#   --ref-load /mnt/share/m00876805/ckpt/Qwen3-30B-MoE
#   --load /mnt/share/m00876805/ckpt/Qwen3-30B-MoE
#   --save /mnt/share/c00937190/ckpt/
#   --save-interval 500
#   --megatron-to-hf-mode bridge
#)


#CKPT_ARGS=(
#   --hf-checkpoint /mnt/share/m00876805/ckpt/Qwen3-30B-MoE
#   --ref-load /mnt/share/c00937190/data/Qwen3-30B-MoE-megatron
#   --load /mnt/share/c00937190/data/Qwen3-30B-MoE-megatron
#   --save /mnt/share/c00937190/ckpt/
#   --save-interval 500
#)


CKPT_ARGS=(
   --hf-checkpoint /mnt/share/m00876805/ckpt/Qwen3-30B-MoE
   --ref-load /mnt/share/c00937190/data/Qwen3-30B-dist
   --load /mnt/share/c00937190/data/Qwen3-30B-dist
   --save /mnt/share/c00937190/ckpt/
   --save-interval 500
)


ROLLOUT_ARGS=(
   --prompt-data /data/rl_slime/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /data/rl_slime/dapo-math-17k/dapo-math-17k.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 8192
   --eval-top-p 0.7
)
 
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 4 8 16 32 64 128 # $(seq 16 8 256)
   # --sglang-disable-cuda-graph
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size 32768
   --sglang-max-prefill-tokens 4000
   --sglang-max-total-tokens 327680
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-attention-backend ascend
   --offload-rollout
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   --use-flash-attn
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8211
sleep 10
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
    \"ASCEND_TOOLKIT_HOME\": \"/usr/local/Ascend/ascend-toolkit/latest/\",
    \"ASCEND_OPP_PATH\": \"/usr/local/Ascend/ascend-toolkit/latest/opp/\",
    \"ASCEND_AICPU_PATH\": \"/usr/local/Ascend/ascend-toolkit/latest/\",
    \"ASCEND_HOME_PATH\": \"/usr/local/Ascend/ascend-toolkit/latest/\",
    \"set_env_path\": \"/usr/local/Ascend/nnal/atb/set_env.sh\",
    \"HYDRA_FULL_ERROR\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8211" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

