#!/bin/bash
# ============================================================
# 模型格式转换: HF → Megatron torch_dist
# ============================================================
# SLIME + Megatron 需要 torch_dist 格式的模型
# 需要先运行这个脚本转换 SFT checkpoint

set -e

SLIME_DIR="${SLIME_DIR:-/root/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"

# ====== 路径配置 ======
HF_CKPT="${1:-/root/Qwen3-1.7B-sft-epoch6}"
OUTPUT_DIR="${2:-/root/Qwen3-1.7B-sft_torch_dist}"

echo "Converting: ${HF_CKPT} → ${OUTPUT_DIR}"

source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

PYTHONPATH=${MEGATRON_DIR} python ${SLIME_DIR}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${HF_CKPT} \
    --save ${OUTPUT_DIR}

echo "Done! Megatron checkpoint saved to: ${OUTPUT_DIR}"
echo ""
echo "Also convert base model (for reference):"
echo "  bash convert_ckpt.sh /root/Qwen3-1.7B /root/Qwen3-1.7B_torch_dist"
