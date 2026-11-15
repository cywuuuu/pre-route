# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional vLLM switches (safe defaults)
os.environ.setdefault("VLLM_ENGINE", "0")
os.environ.setdefault("VLLM_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("VLLM_TEST_DYNAMO_DISABLE", "1")
# ✅ 关键：让 vLLM 用 spawn，而不是 fork
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys
import re
import json
import time
import random
import pickle
import argparse
import hashlib
from typing import List, Dict, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from eval_utils import *
# ------------------------ eval_utils hooks ------------------------
# Expecting these in your environment (as in both original scripts):
#   - get_decision_prompt, select_prompt_builder, expand_example_field,
#     generate_id, compute_and_print_metrics, call_model, get_tokenizer,
#     get_max_workers_for_model, use_thinking_for_model, save_json, load_json
from eval_utils import (
    get_decision_prompt,
    select_prompt_builder,
    expand_example_field,
    generate_id,
    compute_and_print_metrics,
    call_model,
    get_tokenizer,
    get_max_workers_for_model,
    use_thinking_for_model,
    save_json,
    load_json,
)

# --- Optional: tiktoken for token counting (kept) ---
import tiktoken



# ===================== Shared helpers (Code #1) =====================

def load_test_ids(split_indices_path: Optional[str]) -> Optional[set]:
    """Read test_ids from a global_split_indices.json."""
    if not split_indices_path:
        return None
    if not os.path.exists(split_indices_path):
        print(f"[split indices] 未找到文件：{split_indices_path}")
        return None
    try:
        with open(split_indices_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        test_ids = data.get("test_ids", None)
        if not test_ids:
            print(f"[split indices] 文件中未发现 'test_ids' 字段：{split_indices_path}")
            return None
        return set(test_ids)
    except Exception as e:
        print(f"[split indices] 读取失败：{e}")
        return None


def add_ids_to_data(data: List[Dict]) -> List[Dict]:
    """Ensure each example has an _id using generate_id()."""
    for example in data:
        if "_id" not in example:
            example["_id"] = generate_id(example)
    return data


def filter_examples_by_ids(examples: List[Dict], test_ids: set) -> List[Dict]:
    """Filter input examples by test_ids using generate_id()."""
    filtered = []
    for ex in examples:
        try_id = generate_id(ex)
        if try_id in test_ids:
            filtered.append(ex)
        elif "example" in ex and isinstance(ex["example"], dict):
            inner_id = generate_id(ex["example"])
            if inner_id in test_ids:
                filtered.append(ex)
    return filtered


# ===================== Decision text parsers (Code #2) =====================

def extract_reasoning_block(text: str) -> str:
    m = re.search(r"<think>", text, re.IGNORECASE)
    if not m:
        return ""
    tail = text[m.end():]
    m2 = re.search(r"</think>", tail, re.IGNORECASE)
    return (tail[:m2.start()] if m2 else tail).strip()


def extract_thinking_block(text: str) -> str:
    m = re.search(r"<thinking>", text, re.IGNORECASE)
    if not m:
        return text.strip()
    tail = text[m.end():]
    m2 = re.search(r"</thinking>", tail, re.IGNORECASE)
    return (tail[:m2.start()] if m2 else tail).strip()




# ===================== vLLM path (Code #2 style) =====================






def _build_prompts_for_group(group_items: List[Dict], mode: str, prompt_builder, prompt_version: str, head_n: int, input_file: str, tokenizer: AutoTokenizer, rag_chunk_mapping, answer_model: str) -> List[str]:
    prompts = []

    for item in tqdm(group_items, desc="Building prompts", mininterval=10):
        if mode == "ood":
            file_name = item['file']
            query = item['question']
            title = file_name.replace(".txt", "")
            
            rag_chunks = []
            if rag_chunk_mapping is not None:
                key = (file_name, query)
                if key in rag_chunk_mapping:
                    rag_chunks = rag_chunk_mapping[key]


            doc_type = f"{item.get('type', 'unknown')}"
            task_type = f"{item.get('level', 'unknown')}"
            doc_path = f"./datasets/{item['context_length']}/{item['context_type']}/{file_name}"
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
            doc_len_tokens = len(tokenizer.encode(text))
            context_snippet = text.strip()
            answer_model = input_file.split("/")[-2]
            decision_prompt = get_decision_prompt(
                prompt_builder=prompt_builder, prompt_version=prompt_version,
                full_question=query, title=title, doc_type=doc_type,
                doc_len_tokens=doc_len_tokens, context_snippet=context_snippet,
                answer_model=answer_model, task_type=task_type,
                rag_chunks=rag_chunks, head_n=head_n, rag_config=None
            )
        else:
            decision_prompt = item['decision_prompt'].strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": decision_prompt}
        ]
        # enable_thinking if the training item had thinking
        enable_thinking = False if item.get("reason", "") == "" else True

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        prompts.append(prompt)
    return prompts


def run_vllm_router_judge(
    input_json: str,
    router_model_local_cfg: Dict,
    prompt_version: str,
    head_n: int,
    llm,
    rag_chunk_mapping: Dict,
    answer_model: str,
    model_tok,
    mode: str,
    out_path: str,
    test_ids: Optional[set] = None,
    rewrite: bool = False,
) -> Dict:
    """Run the Code #2 (vLLM) style pipeline directly and save to out_path.
    Returns metrics dict produced by compute_and_print_metrics.
    """

    # Load and filter data

    if os.path.exists(out_path) and not rewrite:
        print(f"输出文件已存在，直接复用: {out_path}")
        json_data = load_json(out_path)

        # 若指定 test_ids，只对 test 子集计算指标
        if test_ids:
            json_data = filter_examples_by_ids(json_data, test_ids)
            if not json_data:
                print("[split indices] 警告：现有输出与 test_ids 无交集，无法计算指标。")
                return None, None, None

        # 归并 long_context -> full
        for item in json_data:
            if item.get("route_decision") in ["long_context", "full"]:
                item["route_decision"] = "full"

        json_data = expand_example_field(json_data)
        strict_metrics, lenient_metrics, equal_included_metrics = compute_and_print_metrics(json_data)
        return strict_metrics, lenient_metrics, equal_included_metrics



    data = load_json(input_json)
    if test_ids:
        data = filter_examples_by_ids(data, test_ids)
        if not data:
            print("[split indices] 警告：现有输出与 test_ids 无交集，无法计算指标。")
            return None, None, None
        print(f"[split indices]：现有输出与 test_ids 有交集，共 {len(data)} 条数据。")
    # Group by source (as in Code #2)
    grouped = defaultdict(list)
    for item in data:
        grouped[item.get("source", f"{input_json}")].append(item)

    # Prepare model & tokenizer
    model_path = router_model_local_cfg["model_path"]
    num_gpus = max(1, int(router_model_local_cfg.get("tensor_parallel_size", 1)))



    sampling_params = SamplingParams(
        temperature=1,
        max_tokens=6000,
        stop_token_ids=[model_tok.eos_token_id] if model_tok.eos_token_id is not None else None,
    )

    prompt_builder = select_prompt_builder(prompt_version)

    # Run per source to keep memory steady (mirroring Code #2)
    all_items = []
    for source_key, group_items in grouped.items():
        print(f"[vLLM] source='{source_key}', items={len(group_items)}")
        prompts = _build_prompts_for_group(
            group_items, mode=mode, prompt_builder=prompt_builder,
            prompt_version=prompt_version, head_n=head_n, input_file=input_json, tokenizer=model_tok, rag_chunk_mapping=rag_chunk_mapping, answer_model=answer_model, 
        )
        print(f"[vLLM] Generating {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params)
        print("[vLLM] Generation complete. Parsing...")

        # Merge back
        for item, out in zip(group_items, outputs):
            text = out.outputs[0].text
            reason = extract_reasoning_block(text)
            thinking = extract_thinking_block(text)
            route = extract_decision_from_thinking(thinking)

            item_out = item.copy()
            item_out["thinking_trace"] = thinking
            item_out["route_decision"] = route
            item_out["reason"] = reason
            # unify long_context->full in later metric pipeline
            all_items.append(item_out)

    # Save & compute metrics (same naming convention as Code #1's router_judge)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"[vLLM] Saved results to: {out_path}")

    # Normalize and compute
    for it in all_items:
        if it.get("route_decision") in ["long_context", "full"]:
            it["route_decision"] = "full"
    metrics = compute_and_print_metrics(all_items)
    return metrics


# ===================== Code #1-style file path helpers =====================

def _get_input_jsons(answer_model: str, compare_model: str) -> list:
    thinking_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_thinking_comparison.json"
    instruct_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_comparison.json"
    input_jsons = []
    if os.path.exists(thinking_path):
        input_jsons.append(thinking_path)
    if os.path.exists(instruct_path):
        input_jsons.append(instruct_path)
    return input_jsons


def _router_judge_output_path(input_json: str, router_model: str, prompt_version: str, best_of_n: int, router_thinking: bool) -> str:
    return input_json.replace(
        ".json",
        f"_router{router_model}_{prompt_version}_bon{best_of_n}_thinking{router_thinking}.json"
    )


def enumerate_all_tasks(args, update_rag: bool = True) -> list:
    tasks = []
    for answer_model in args.answer_models:
        for compare_model in args.compare_models:
            input_jsons = _get_input_jsons(answer_model, compare_model)
            if not input_jsons:
                continue
            for router_model in args.router_models:
                thinkings = use_thinking_for_model(router_model) if 'use_thinking_for_model' in globals() else [False]
                for input_json in input_jsons:
                    for router_thinking in thinkings:
                        # router_judge
                        prompt_version = args.prompt_version
                        out_main = _router_judge_output_path(input_json, router_model, prompt_version, args.best_of_n, router_thinking)
                        tasks.append({
                            "type": "router_judge",
                            "answer_model": answer_model,
                            "compare_model": compare_model,
                            "router_model": router_model,
                            "input_json": input_json,
                            "router_thinking": router_thinking,
                            "prompt_version": prompt_version,
                            "out_path": out_main,
                        })
    return tasks


def summarize_task_progress(args, update_rag: bool = True, show_missing_examples: int = 0) -> dict:
    tasks = enumerate_all_tasks(args, update_rag=update_rag)
    total = len(tasks)
    done = sum(1 for t in tasks if os.path.exists(t["out_path"]))
    remaining = total - done

    by_type = defaultdict(lambda: {"total": 0, "done": 0})
    for t in tasks:
        by_type[t["type"]]["total"] += 1
        if os.path.exists(t["out_path"]):
            by_type[t["type"]]["done"] += 1

    print("\n========== 任务进度统计 ==========")
    print(f"总任务数: {total}")
    print(f"已完成 : {done}")
    print(f"未完成 : {remaining}")
    for k, v in by_type.items():
        print(f"- {k}: {v['done']}/{v['total']} 已完成")

    if show_missing_examples > 0 and remaining > 0:
        print("\n未完成样例（仅显示前 {} 条）：".format(show_missing_examples))
        count = 0
        for t in tasks:
            if not os.path.exists(t["out_path"]):
                count += 1
                tag = f"{t['type']} | router={t['router_model']} | answer={t['answer_model']} | compare={t['compare_model']} | thinking={t['router_thinking']}"
                if t["type"] == "router_judge":
                    tag += f" | prompt={t['prompt_version']}"
                print(f"  - {tag}")
                print(f"    输入: {t['input_json']}")
                print(f"    输出: {t['out_path']}")
                if count >= show_missing_examples:
                    break

    summary = {
        "total": total,
        "done": done,
        "remaining": remaining,
        "by_type": {k: {"total": v["total"], "done": v["done"], "remaining": v["total"] - v["done"]} for k, v in by_type.items()},
        "tasks": tasks,
    }
    print("=================================\n")
    return summary

# checkpoint produced by train_router_sft.py (step 5, default --tag/--epochs);
# update this path if you trained with a different tag or epoch count
model2path = {
    "local_sft_qwen3_1_7b": "outputs/Qwen3-1.7B_sft_default/checkpoints/epoch-006",
}
# ===================== CLI & main routine =====================

def parse_args():
    parser = argparse.ArgumentParser(description="路由评估脚本（vLLM 直接调用）")
    parser.add_argument("--answer_models", nargs='+', default=[
        "qwen3-235b-a22b-instruct-2507",
        "qwen3-235b-a22b-thinking-2507",
        "deepseek-r1-0528",
        "qwen-max-latest",
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen3-30b-a3b-instruct-2507",
        "qwen3-30b-a3b-thinking-2507",
    ])
    parser.add_argument("--router_models", nargs='+', default=[
        "local_sft_qwen3_1_7b"
    ])
    parser.add_argument("--compare_models", nargs='+', default=[
        "qwen3-235b-a22b-instruct-2507"
    ])

    parser.add_argument("--prompt_versions", default=["preroute"], nargs='+',
                        choices=["preroute"])

    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--head_n", type=int, default=2000)

    # Sampling / cache options preserved from Code #1
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--sample_seed", type=int, default=42)

    # Test split filtering
    parser.add_argument("--split_indices_path", type=str,
                        default="distillation_datasets/from_multi_answer_models_0.7_0.1_0.2_42/global_split_indices.json")

    # vLLM local SFT options (for router_model == local_sft_qwen3_1_7b)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override automatic local SFT model path for vLLM")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU devices to use for tensor parallelism, e.g. '0,1'")
    parser.add_argument("--mode", type=str, choices=["standard", "ood"], default="ood")

    parser.add_argument("--rewrite", action="store_true", default=False,
                        help="是否强制重写输出文件")
    parser.add_argument('--ragchunk_path', type=str, default=None, help='pkl file with rag chunk cache')
    return parser.parse_args()


def main():
    args = parse_args()

    for prompt_version in args.prompt_versions:
        args.prompt_version = prompt_version
        # Load test ids if any
        test_ids = load_test_ids(args.split_indices_path)
        if test_ids:
            print(f"[split indices] 已加载 test_ids 数量：{len(test_ids)}，将按 test_ids 评测。")
        else:
            print("[split indices] 未启用或加载失败，保持现有评测/采样流程。")
        ragchunk_path = args.ragchunk_path or f"./datasets/all_query_map_ragchunks.pkl"
        # Load RAG chunk cache (kept for compatibility; not directly used in vLLM path)
        if os.path.exists(ragchunk_path):
            print(f"加载RAG缓存: {ragchunk_path}")
            with open(ragchunk_path, "rb") as f:
                rag_chunk_mapping = pickle.load(f)
            print(f"RAG缓存条目数: {len(rag_chunk_mapping)}")
        else:
            print(f"未找到RAG缓存 {ragchunk_path}，后续全部实时retrieval！")
            rag_chunk_mapping = {}

        # Determine temperature by BoN (kept semantics)
        if args.best_of_n <= 0:
            temperature = 0
            args.best_of_n = 1
            print(f"Using Greedy Search, setting temperature to {temperature}")
        else:
            temperature = 1
            print(f"Using BoN= {args.best_of_n}, setting temperature to {temperature}")

        # Enumerate tasks & show progress snapshot
        _ = summarize_task_progress(args, update_rag=True, show_missing_examples=5)

        # Build list of input_jsons to run (mirroring Code #1)
        all_input_jsons = []
        for compare_model in args.compare_models:
            for answer_model in args.answer_models:
                paths = _get_input_jsons(answer_model, compare_model)
                all_input_jsons.extend(paths)
        all_input_jsons = list(dict.fromkeys(all_input_jsons))  # dedup

        # Ensure _id in each input json
        for input_json in all_input_jsons:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and "_id" not in data[0]:
                data = add_ids_to_data(data)
                with open(input_json, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        out_metrics_path = f"evaluation/prediction/lara_router_metrics_{prompt_version}_{bool(args.split_indices_path)}.json"
        # if os.path.exists(out_metrics_path):
        #     # read
        #     with open(out_metrics_path, "r", encoding='utf-8') as f:
        #         res_metrics = json.load(f)
        # else:
        # Traverse like Code #1, but dispatch to vLLM or fallback per router_model
        res_metrics = []
        total_out_metrics_path = out_metrics_path.replace(".json", "_local.json")
        for router_model in args.router_models:



            # Prepare local SFT model path for vLLM if needed
            router_local_cfg = None

            model_path = args.model_path or model2path[router_model]

            tp_size = max(1, len(args.gpu.split(',')))
            router_local_cfg = {"model_path": model_path, "tensor_parallel_size": tp_size}
            print(f"[vLLM] Local SFT router path prepared: {model_path} (TP={tp_size})")
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=router_local_cfg['tensor_parallel_size'],
                max_model_len=24000
            )
            model_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            for answer_model in args.answer_models:
                for compare_model in args.compare_models:
                    # discover files again for this pair
                    input_jsons = _get_input_jsons(answer_model, compare_model)
                    if not input_jsons:
                        continue
                    thinkings = use_thinking_for_model(router_model)
                    for input_json in input_jsons:
                        for router_thinking in thinkings:
                            max_workers = int(get_max_workers_for_model(router_model) * 1.2)

                            out_path = _router_judge_output_path(input_json, router_model, prompt_version, args.best_of_n, router_thinking)

                            result_metrics_item = {
                                "router_model": router_model,
                                "compare_model": compare_model,
                                "answer_model": answer_model,
                                "router_thinking": router_thinking,
                                "answer_thinking": "thinking_comparison.json" in input_json,
                                "prompt_version": prompt_version,
                                "metric": {}
                            }

                            # Prefer vLLM path when router_model is local SFT and vLLM is available

                            print("==== router_judge via vLLM (direct) ====")
                            print(f"input: {input_json}\nrouter_model: {router_model}\nprompt: {prompt_version}\nTP: {router_local_cfg['tensor_parallel_size']}")
                            metrics = run_vllm_router_judge(
                                input_json=input_json,
                                router_model_local_cfg=router_local_cfg,
                                prompt_version=prompt_version,
                                llm=llm,
                                model_tok=model_tok,
                                head_n=args.head_n,
                                mode=args.mode,
                                out_path=out_path,
                                test_ids=test_ids,
                                rewrite=args.rewrite,
                                rag_chunk_mapping=rag_chunk_mapping,
                                answer_model=answer_model,
                            )
                            # 修改为扁平结构，与3router_judge_sample_bon.py保持一致
                            result_metrics_item.update({
                                "kind": "router_judge",
                                "best_of_n": int(args.best_of_n),
                                "strict": metrics[0] if isinstance(metrics, (list, tuple)) else metrics,
                                "lenient": metrics[1] if isinstance(metrics, (list, tuple)) and len(metrics) > 1 else None,
                                "equal_included": metrics[2] if isinstance(metrics, (list, tuple)) and len(metrics) > 2 else None,
                            })
                            # 移除嵌套的metric字段
                            del result_metrics_item["metric"]

                            res_metrics.append(result_metrics_item)

            with open(total_out_metrics_path, "w", encoding='utf-8') as f:
                json.dump(res_metrics, f, indent=2, ensure_ascii=False)
            print(f"[保存] 指标到 {total_out_metrics_path}")


if __name__ == "__main__":
    main()
