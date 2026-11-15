# Pre-Route

Proactive routing between **Retrieval-Augmented Generation (RAG)** and **Long-Context (LC)**: before answering, a router performs lightweight structured reasoning over query/document metadata and decides which path to take — cutting LC cost while keeping answer quality. The routing capability can be elicited in large models with structured guidelines, and distilled into a 1.7B router for lightweight deployment.


## Repository layout

```
evaluation/          # the full pipeline: from raw LaRA queries to a distilled router
  ├── 1eval_full.py                      # step 1a: run the long-context path
  ├── 1eval_rag.py                       # step 1b: run the RAG path
  ├── 2merge_compare.py                  # step 2: merge both paths + judge-compare
  ├── 3router_judge_sample_bon.py        # step 3: routing inference + evaluation
  ├── sft1_split_data.py                 # step 4: 70/10/20 split + RFT teacher-consistency filtering
  ├── train_router_sft.py                # step 5: RFT router distillation training
  ├── sft3_router_single_vllm.py         # step 6: vLLM inference for the distilled router
  ├── prompt.py                          # prompt templates, incl. Pre-Route guidelines
  ├── eval_utils.py                      # shared utilities: prompt building, API calls, scoring
  ├── qa_score_evaluator.py              # QA scoring, 4-point rubric
  ├── rag_chunks_all.py                  # optional: regenerate the RAG chunk cache
  └── search/                            # hybrid retrieval for rag_chunks_all
datasets/             # LaRA data
opd/                  # optional add-on: on-policy distillation
```

This release covers the LaRA pipeline — routing inference and evaluation, and RFT distillation.

## Setup

```bash
pip install -r requirements.txt
```

**Models** — used for tokenization and local inference:
```bash
mkdir -p model_weights
# download and place:
#   Qwen/Qwen3-1.7B        -> model_weights/Qwen3-1.7B
```

**Embedding models** — only needed to regenerate the RAG chunk cache; a precomputed cache ships in `datasets/`:
```bash
mkdir -p embedding_models
# download from Hugging Face and place:
#   Alibaba-NLP/gte-multilingual-base           -> embedding_models/gte-multilingual-base
#   Alibaba-NLP/gte-multilingual-reranker-base  -> embedding_models/gte-multilingual-reranker-base
```

**API keys** — the router/answer/judge models are called through provider APIs; set the ones you use:
```bash
export DASHSCOPE_API_KEY=...        # DashScope / Aliyun
export OPENAI_API_KEY=...           # OpenAI
export OPENROUTER_API_KEY=...       # OpenRouter
export GOOGLE_API_KEY=...           # Google
```

## Steps 1–3 — routing evaluation with a large model

```bash
# 1. run both answer paths on the bundled 15-query demo cell
python evaluation/1eval_full.py --limit 15
python evaluation/1eval_rag.py  --limit 15

# 2. merge the two paths and judge-compare them
python evaluation/2merge_compare.py

# 3. run the router over the comparison file
python evaluation/3router_judge_sample_bon.py \
    --answer_models qwen3-30b-a3b-instruct-2507 \
    --router_models qwen3-235b-a22b-instruct-2507 \
    --compare_models qwen3-235b-a22b-instruct-2507 \
    --prompt_versions preroute
```

- `--prompt_versions preroute` is the full Pre-Route guideline prompt.
- Step 3 reads `evaluation/prediction/<answer_model>/flattened_merged_predictions_<compare_model>_comparison.json` produced by step 2; the `--judge_model` of step 2 must equal the `--compare_models` of step 3. It writes per-entry routing records plus `evaluation/prediction/lara_router_metrics_<...>.json` with **QA score**, **LC selection rate** and **Route Accuracy**. The ideal label is the path with higher QA performance; ties default to RAG.
- Without `--split_indices_path`, all queries are evaluated.
- QA rubric scoring: `python evaluation/qa_score_evaluator.py`.

## Steps 4–6 — distill your own router (RFT)

```bash
# 4. stratified split + teacher-consistency filtering -> distillation_datasets/
#    input_files at the top of sft1_split_data.py defaults to the demo chain output
python evaluation/sft1_split_data.py

# 5. RFT training
python evaluation/train_router_sft.py --epochs 6

# 6. run your distilled router
python evaluation/sft3_router_single_vllm.py
```

Repeat steps 1–3 across query cells to scale up the distillation data. All data construction calls answer/judge/router APIs; the 15-query demo costs a few cents.

## OPD (optional add-on)

`opd/` is an optional add-on: an on-policy distillation setup built on **SLIME + Megatron-LM** with a locally served teacher. Requires docker and GPU infrastructure — see `opd/OPD_GUIDE.md`. For most users, the RFT path above is sufficient.

## Data

`datasets/` holds the LaRA benchmark data in the layout the pipeline expects:

- `datasets/query/{32k|128k}_{financial|paper|book}_{hallu|location|reasoning|comp}.jsonl` — queries; the full benchmark has 24 files and 2,326 cases
- `datasets/{32k,128k}/<type>/` — source documents
- `datasets/all_query_map_ragchunks.pkl` — precomputed RAG chunk cache; 26 MB, covers all LaRA queries

The bundled demo slice is 32k/financial/hallu: 15 queries plus their 4 source documents, so steps 1–3 run for a few cents. The demo is for verifying the pipeline; expect small-sample variance. LaRA queries have free-text gold answers; per-path quality is scored 1–4 by a judge model — rubric in `qa_score_evaluator.py`.

### Running the full benchmark

1. Get the full data — clone the official LaRA release, arXiv 2502.09977, and copy its `datasets/` over this repo's. The layout is the same; the full `32k_financial_hallu.jsonl` supersedes the bundled slice:

```bash
git clone https://github.com/Alibaba-NLP/LaRA /tmp/LaRA
cp -r /tmp/LaRA/datasets/* datasets/
```

The bundled chunk cache covers all 2,326 queries, so the RAG path needs no retrieval setup. Documents must sit at `datasets/{32k,128k}/<type>/<file>`, matching the `file` field of each query; if your copy stores them flat, arrange them into the per-type subfolders first.

2. Run both answer paths over all 24 query cells. Each call processes one cell; existing outputs are skipped, so the loop is resumable — add `--limit N` for a trial:

```bash
for L in 32k 128k; do for C in financial paper book; do for Q in hallu location reasoning comp; do
    python evaluation/1eval_full.py --context_length $L --context_type $C --query_type $Q
    python evaluation/1eval_rag.py  --context_length $L --context_type $C --query_type $Q
done; done; done
```

3. Steps 2–6 are unchanged: `2merge_compare.py` merges every prediction file of the answer model at once.

Full-benchmark cost: 2,326 queries, each with two answer calls plus one judge call and one router call — on the order of a few USD depending on the models chosen.

## License

Apache-2.0
