#!/bin/bash
# ============================================================
# SLIME OPD: Qwen3-1.7B Router (teacher = SGLang-style /generate)
# ============================================================
# 使用方式 (两种 teacher 接法, 详见 OPD_GUIDE.md):
#   1. 起 teacher:
#      推荐: python -m sglang.launch_server --model-path <teacher-model> --port 30000
#            并把下方 TEACHER_URL 改为 http://127.0.0.1:30000/generate
#      (或:  云端 proxy — bash run_proxy.sh, 现状限制见 OPD_GUIDE.md)
#   2. 再运行本脚本:          bash run-qwen3-1.7B-opd-bailian.sh
#
# 前置条件:
#   - SLIME 已安装 (在 docker 中)
#   - SFT checkpoint 已转换为 Megatron torch_dist 格式
#   - teacher 服务已启动 (sglang server 或 proxy)
# ============================================================

set -ex

export PYTHONBUFFERED=16

# ====== 路径配置 (按实际修改) ======
SLIME_DIR="/root/slime"
MEGATRON_DIR="/root/Megatron-LM"

# Student 模型 (SFT 后的 Qwen3-1.7B)
HF_CKPT="/root/Qwen3-1.7B-sft-epoch6"          # HF 格式 checkpoint
TORCH_DIST_CKPT="/root/Qwen3-1.7B-sft_torch_dist"  # Megatron 格式
SAVE_DIR="/root/Qwen3-1.7B-opd"                     # OPD 训练保存路径

# Reference 模型 (通常 = SFT checkpoint 本身)
REF_LOAD="${TORCH_DIST_CKPT}"

# 数据
TRAIN_DATA="/root/opd_data/lara_train_prompts.jsonl"
# 可选: 如需训练中评估, 自行接入 eval prompt 文件 (如 lara_eval_prompts.jsonl, 见 SLIME --eval-* 参数)

# Teacher 地址 (SGLang server 或 proxy)
TEACHER_URL="http://127.0.0.1:13141/generate"

# GPU 配置
NUM_GPUS=${NUM_GPUS:-4}

# ====== Qwen3-1.7B 模型参数 ======
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

# ====== Checkpoint ======
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   --load ${SAVE_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

# ====== Rollout ======
ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

# ====== OPD Reward (Teacher) ======
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url ${TEACHER_URL}
)

# ====== Eval (可选) ======
# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data <dataset-name> <eval-prompts.jsonl>
#    --n-samples-per-eval-prompt 4
#    --eval-max-response-len 4096
# )

# ====== Performance ======
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

# ====== GRPO + OPD ======
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ====== Optimizer ======
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ====== SGLang (Rollout Engine) ======
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
)

# ====== Misc ======
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ====== Wandb (可选) ======
# WANDB_ARGS=(
#    --use-wandb
#    --wandb-project pre-route-opd
#    --wandb-group qwen3-1.7B-opd
#    --wandb-key ${WANDB_KEY}
# )

# ====== Launch ======
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=1
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

# 清理
pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 python
