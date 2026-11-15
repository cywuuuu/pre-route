# OPD（On-Policy Distillation）附加包指南

## 这是什么

在 RFT 之后的**可选进阶训练**：学生路由器（SFT 后的 Qwen3-1.7B）自行生成 routing rollout，
teacher 模型对每条 rollout 给出逐 token log-probs，SLIME 在 GRPO 训练中用其做 on-policy 蒸馏信号。

- OPD 核心逻辑（`slime/rollout/on_policy_distillation`）是**官方 SLIME 自带**的，本目录不含自研训练代码；
  这里提供的是：训练启动脚本、权重格式转换、prompt 数据，以及一个可选的云端 teacher proxy。
- 门槛：SLIME + Megatron-LM docker 环境 + 多卡 GPU。大多数用户用 README 的 RFT 路线（steps 4-6）即可。

## 目录内容

| 文件 | 作用 |
|------|------|
| `run-qwen3-1.7B-opd-bailian.sh` | SLIME OPD 训练启动脚本（**teacher 无关**：任何 SGLang 风格 `/generate` 端点均可；文件名里的 bailian 只是历史命名） |
| `convert_ckpt.sh` | HF checkpoint → Megatron torch_dist 格式转换 |
| `lara_train_prompts.jsonl` (6914) / `lara_eval_prompts.jsonl` (1356) | SLIME `--prompt-data` 训练 / 训练中评估数据 |
| `bailian_teacher_proxy.py` + `run_proxy.sh` | 可选云端 teacher proxy（现状见文末） |

prompt 格式：每行 `{"prompt": [{"role":"system","content":"..."}, {"role":"user","content":"<decision_prompt>"}]}`，SLIME 加 `--apply-chat-template` 直接使用。

## 操作步骤

### 0. 前置

- SLIME 官方 docker（建议固定版本——官方仍在持续修复 OPD 相关问题）+ Megatron-LM
- 学生 checkpoint = RFT step 5 产出，如 `outputs/Qwen3-1.7B_sft_default/checkpoints/epoch-006`

### 1. 转换模型格式

SLIME + Megatron 需要 `torch_dist` 格式：

```bash
bash opd/convert_ckpt.sh <hf_ckpt> <torch_dist_output>
# 例: bash opd/convert_ckpt.sh outputs/Qwen3-1.7B_sft_default/checkpoints/epoch-006 /root/Qwen3-1.7B-sft_torch_dist
```

### 2. 启动 Teacher（推荐：本地 SGLang 起大模型）

```bash
python -m sglang.launch_server --model-path <一个大模型，如 Qwen3-235B/Qwen3-32B 本地权重> --port 30000
```

然后编辑 `run-qwen3-1.7B-opd-bailian.sh`：

```bash
TEACHER_URL="http://127.0.0.1:30000/generate"
```

SLIME 的 `--opd-type sglang` 原生对接 SGLang 的 `/generate`（返回 input_token_logprobs），**无需任何适配代码**。
前提：teacher 与 student 同源 tokenizer（Qwen 系列 vocab 一致）。

### 3. 运行 OPD 训练

编辑 run 脚本里的各路径（SLIME_DIR / 权重 / 数据）为实际值后：

```bash
bash opd/run-qwen3-1.7B-opd-bailian.sh
```

### 4. 评估 OPD 后的模型

用 SLIME 自带的转换工具把保存的 torch_dist checkpoint 转回 HF 格式，然后用本仓库的评估脚本：

```bash
python evaluation/sft3_router_single_vllm.py --model_path <opd-hf-checkpoint> ...
```

（评估脚本的 API 调用统一走 `eval_utils.py`，key 设置见 README。）

---

## 可选：云端 teacher proxy（bailian，当前受限）

`bailian_teacher_proxy.py` 用 FastAPI 模拟 SGLang 的 `/generate` 接口，收到 token IDs 后解码为文本，
转调 DashScope OpenAI 兼容层的 `/completions`（`echo=True` + `logprobs`）拿 prompt 侧 log-probs 再转回 SGLang 格式。

```bash
export BAILIAN_API_KEY="sk-xxx"
bash opd/run_proxy.sh          # 默认 127.0.0.1:13141，TEACHER_URL 指向 http://127.0.0.1:13141/generate
```

> ⚠️ **现状说明（2026-11 实测）**：DashScope 兼容层 `/completions` 端点对 `qwen-max`/`qwen-plus` 等返回
> `model_not_supported`，该端点的可用模型集合与能力会随时间变化——**此路线当前不可用，请以本地 SGLang
> teacher 为准**；如仍想尝试云端方案，请先自行验证该端点对所选模型是否开放 `echo` + `logprobs`。

---

## 数据流概要

```
                    SLIME OPD 训练
                    ─────────────
LaRA prompt ──→ Student (SFT 1.7B) ──→ 生成 routing 决策 (rollout)
 (6914条)           ↓                         ↓
               Teacher (SGLang server 或 proxy) ──→ log-probs
                    ↓                         ↓
               OPD KL Loss ←──── 对比 student vs teacher log-probs
                    ↓
               更新 Student 模型

                    评估
                    ──
OPD 后模型 ──→ vLLM 推理 ──→ evaluation/sft3_router_single_vllm.py
```
