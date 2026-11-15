#!/bin/bash
# ============================================================
# 启动 Bailian Teacher Proxy
# ============================================================
# 用法: bash run_proxy.sh
# 前台运行, 另开终端运行 SLIME 训练脚本

set -e

# ====== 配置 ======
export BAILIAN_API_KEY="${BAILIAN_API_KEY:-}"          # 替换为你的 API Key
TEACHER_MODEL="${TEACHER_MODEL:-qwen-max}"              # 百炼模型
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/Qwen3-1.7B-sft-epoch6}"  # Tokenizer 路径
PORT="${PORT:-13141}"

if [ -z "$BAILIAN_API_KEY" ]; then
    echo "ERROR: 请设置 BAILIAN_API_KEY 环境变量"
    echo "  export BAILIAN_API_KEY=sk-xxxxx"
    exit 1
fi

echo "Starting Bailian Teacher Proxy..."
echo "  Model:  ${TEACHER_MODEL}"
echo "  Port:   ${PORT}"
echo "  Tokenizer: ${TOKENIZER_PATH}"

python bailian_teacher_proxy.py \
    --port ${PORT} \
    --model ${TEACHER_MODEL} \
    --tokenizer-path ${TOKENIZER_PATH}
