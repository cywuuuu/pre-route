import json
import re
import argparse
import random
import os
from typing import List, Dict, Optional
import tiktoken
from eval_utils import *
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
import pickle
import time 
import shutil
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_test_ids(split_indices_path: Optional[str]) -> Optional[set]:
    """从 global_split_indices.json 读取 test_ids。json 结构需包含 'test_ids'。"""
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


def filter_examples_by_ids(examples: List[Dict], test_ids: set) -> List[Dict]:
    """
    对输入 examples（来自 input_json）按 test_ids 进行筛选。
    若样本还没有 _id，这里会用 generate_id(example) 生成后再筛。
    """
    filtered = []
    for ex in examples:
        if generate_id(ex) in test_ids:
            filtered.append(ex)
        elif "example" in ex and generate_id(ex["example"]) in test_ids:
            filtered.append(ex)
    return filtered


def filter_results_by_ids(results: List[Dict], test_ids: set) -> List[Dict]:
    """
    通用过滤函数：对结果列表按 test_ids 筛选。
    支持 router_judge 结果结构：
      - router_judge: {"example_id": "..."} 或 {"example": {"_id": "..."}}
    """
    out = []
    for r in results:
        example_id = None
        # 优先取顶层的 example_id
        if "example_id" in r:
            example_id = r["example_id"]
        # 否则尝试取嵌套的 example._id
        elif "example" in r and isinstance(r["example"], dict):
            example_id = r["example"].get("_id")

        if example_id in test_ids:
            out.append(r)
    return out
def add_ids_to_data(data: List[Dict]) -> List[Dict]:
    """为数据集中的每个样本添加_id"""
    for example in data:
        if "_id" not in example:
            example["_id"] = generate_id(example)
    return data
def compute_metrics(results: List[Dict]) -> Dict:
    y_true = [1 if result["better_strategy"] == "rag" else 0 for result in results]
    y_pred = [1 if result["route_decision"] == "rag" else 0 for result in results]
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": accuracy,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        },
        "total_examples": len(results)
    }
from collections import defaultdict
from typing import Dict, List, Optional

def count_level_type_ctx_distribution(
    json_data: List[Dict],
    type_merge_map: Optional[Dict[str, str]] = None,
    level_key: str = "level",
    type_key: str = "type",
    ctx_key: str = "context_length"
) -> Dict:
    """
    统计 Level × Type × Context-Length 分布。
    - json_data: 样本列表；每个 item 最终要能展开为包含 level/type/context_length 的字典。
    - type_merge_map: 可选的类型合并映射，如 {"book":"Novel", "novel":"Novel", ...}
    - *_key: 三个字段名可自定义。

    返回:
        {
          "counts": {level: {type: {ctx_len: count}}}
          "total_count": int,
          "levels": [..], "types": [..], "ctx_lens": [..],
          "summaries": {
              "by_level": {level: total},
              "by_type": {type: total},
              "by_ctx": {ctx: total},
              "by_level_type": {level: {type: total}},
              "by_level_ctx": {level: {ctx: total}},
              "by_type_ctx": {type: {ctx: total}}
          }
        }
    """
    # 如你之前的 expand 函数
    expanded = expand_example_field(json_data)

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_count = 0

    for it in expanded:
        lvl = str(it.get(level_key, "unknown"))
        typ_raw = str(it.get(type_key, "unknown"))
        ctx = str(it.get(ctx_key, "unknown"))

        typ = type_merge_map.get(typ_raw, typ_raw) if type_merge_map else typ_raw

        counts[lvl][typ][ctx] += 1
        total_count += 1

    # 规范排序
    levels = sorted(counts.keys())
    types = sorted({t for lv in counts.values() for t in lv.keys()})
    # 按 32k/128k 优先，其它在后
    ctx_all = sorted({c for lv in counts.values() for tp in lv.values() for c in tp.keys()})
    ctx_order = [c for c in ("32k", "128k") if c in ctx_all] + [c for c in ctx_all if c not in ("32k","128k")]

    # 汇总
    by_level = {lv: sum(sum(tp.values()) for tp in counts[lv].values()) for lv in levels}
    by_type = {tp: sum(counts[lv][tp].get(c, 0) for lv in levels for c in ctx_order) for tp in types}
    by_ctx  = {c: sum(counts[lv][tp].get(c, 0) for lv in levels for tp in types) for c in ctx_order}

    by_level_type = {
        lv: {tp: sum(counts[lv][tp].get(c, 0) for c in ctx_order) for tp in types}
        for lv in levels
    }
    by_level_ctx = {
        lv: {c: sum(counts[lv][tp].get(c, 0) for tp in types) for c in ctx_order}
        for lv in levels
    }
    by_type_ctx = {
        tp: {c: sum(counts[lv][tp].get(c, 0) for lv in levels) for c in ctx_order}
        for tp in types
    }

    return {
        "counts": {lv: {tp: dict(counts[lv][tp]) for tp in counts[lv]} for lv in counts},
        "total_count": total_count,
        "levels": levels,
        "types": types,
        "ctx_lens": ctx_order,
        "summaries": {
            "by_level": by_level,
            "by_type": by_type,
            "by_ctx": by_ctx,
            "by_level_type": by_level_type,
            "by_level_ctx": by_level_ctx,
            "by_type_ctx": by_type_ctx,
        }
    }
def count_type_level_distribution(json_data: List[Dict]) -> Dict:
    """
    统计json_data中每个item["example"]的type X level的个数分布
    
    Args:
        json_data: 包含example字段的数据列表
        
    Returns:
        包含type X level分布统计的字典
    """
    # 展开example字段
    expanded_data = expand_example_field(json_data)
    
    # 使用defaultdict来统计type X level的分布
    type_level_counts = defaultdict(lambda: defaultdict(int))
    total_count = 0
    
    for item in expanded_data:
        # 获取type和level，如果没有则使用"unknown"
        item_type = item.get("type", "unknown")
        item_level = item.get("level", "unknown")
        
        # 统计type X level的个数
        type_level_counts[item_type][item_level] += 1
        total_count += 1
    
    # 转换为普通字典并添加汇总信息
    result = {
        "distribution": {k: dict(v) for k, v in type_level_counts.items()},
        "total_count": total_count,
        "type_summary": {k: sum(v.values()) for k, v in type_level_counts.items()},
        "level_summary": defaultdict(int)
    }
    
    # 计算level的汇总统计
    for type_dict in type_level_counts.values():
        for level, count in type_dict.items():
            result["level_summary"][level] += count
    
    result["level_summary"] = dict(result["level_summary"])
    
    return result

def print_ctx_type_level_distribution(stats: Dict):
    """
    按 context_length 分类打印 Type × Level 二维表
    Args:
        stats: count_level_type_ctx_distribution 返回的结果
    """
    levels = stats["levels"]
    types = stats["types"]
    ctxs  = stats["ctx_lens"]
    counts = stats["counts"]

    print("\n=== Context × Type × Level 分布统计 ===")
    print(f"总样本数: {stats['total_count']}")

    # 遍历每个 context_length
    for ctx in ctxs:
        print(f"\n--- Context Length: {ctx} ---")
        header = ["Type"] + levels + ["Total"]
        line_w = sum(len(h)+2 for h in header)
        print("\t".join(header))
        print("-" * line_w)

        # 每个 type 一行
        for tp in types:
            row_vals = [counts.get(lv, {}).get(tp, {}).get(ctx, 0) for lv in levels]
            row_total = sum(row_vals)
            print("\t".join([tp] + [str(v) for v in row_vals] + [str(row_total)]))

        # 本 context 的列合计
        col_totals = [stats["summaries"]["by_level_ctx"][lv].get(ctx, 0) for lv in levels]
        print("-" * line_w)
        print("\t".join(["(Ctx total)"] + [str(v) for v in col_totals] + [str(sum(col_totals))]))

def process_item(example: Dict, router_model: str, answer_model: str, best_of_n: int = 1, temperature: float = 0.0, prompt_version: str = "v2", head_n = 2000, rag_chunk_mapping: dict = {}, thinking: bool = False) -> Optional[Dict]:
    file_name = example['file']
    full_question = example['question']
    title = file_name.replace(".txt", "")
    tokenizer = get_tokenizer(answer_model)
    model = answer_model
    doc_type = f'{example.get("type", "unknown")}'
    task_type = f'{example.get("level", "unknown")}'
    with open(f'./datasets/{example["context_length"]}/{example["context_type"]}/{file_name}', 'r', encoding='utf-8') as f:
        context_snippet = f.read()
    doc_len_tokens = len(tokenizer.encode(context_snippet))
    
    # doc_head = text.strip()[:2000]
    rag_chunks = []
    if rag_chunk_mapping is not None:
        key = (file_name, full_question)
        if key in rag_chunk_mapping:
            rag_chunks = rag_chunk_mapping[key]
    if isinstance(rag_chunks, str):
        # 按 "chunk <数字>:" 拆分
        rag_chunks = re.split(r"chunk\s+\d+:", rag_chunks)
        # 去掉空的段落
        rag_chunks = [c.strip() for c in rag_chunks if c.strip()]

    

        prompt_builder = select_prompt_builder(prompt_version)
        decision_prompt = get_decision_prompt(
            prompt_builder, prompt_version, full_question, title, doc_type, 
            doc_len_tokens, context_snippet, answer_model, task_type, 
            rag_chunks, head_n, rag_config=None
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": decision_prompt}
    ]
    token_len = len(tokenizer.encode(decision_prompt))
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # # ===
    # import dashscope
    # # 请确保 dashscope.api_key 已在别处设置，或在这里设置
    # dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    # resp = dashscope.Tokenization.call(
    #     model="qwen-max",
    #     messages=messages
    # )

    # print(f"输入tokens: {len(tokenizer.encode(text))} || resp.usage[\"input_tokens\"]: {resp.usage}" )

    # # ===
    # # 再对这个拼接好的字符串进行分词，统计 token 数

    routes = None
    thinkings = None

    for _ in range(3):
        try:
            reason, reply = call_model(messages, eval_model=router_model, thinking=thinking, best_of_n=best_of_n, temperature=temperature)
            if not reply:
                continue
            
            thinkings = []
            routes = []
            if isinstance(reply, list):
                # 处理 BoN 返回的多个结果
                for r in reply:

                    thinking_block = extract_thinking_block(r)
                    thinkings.append(thinking_block)
                    route_decision = extract_decision_from_thinking(thinking_block)
                    
                    if not route_decision or route_decision == "no_matching":
                        print(f"❌ 无效的路由决策: {example['_id']}, thinking: {thinking_block}")
                        continue
                    
                    if route_decision in {"rag", "full", "long_context"}:
                        routes.append(route_decision)
                
                if routes:  # 如果有有效的路由决策，则跳出重试循环
                    break
            else:
                # 处理单个结果
                thinking = extract_thinking_block(reply)
                route = extract_decision_from_thinking(thinking)
                
                if not route or route == "no_matching":
                    print(f"❌ 无有效路由决策: {example['_id']}")
                else:
                    if route in {"full", "long_context"}:
                        route = "full"  # 统一处理为 full
                    
                    routes.append(route)
                    thinkings.append(thinking)
                    
                    if route in {"rag", "full", "long_context"}:
                        break
        except Exception as e:
            print(f"[Strategy Retry] {e}")
            routes = None
            thinkings = None
            time.sleep(1)
    
    # 如果没有有效的路由决策，返回 None
    if not routes:
        print(f"Failed to get a valid route for {file_name}")
        return None
    

    
    
    better_strategy = "rag" if example["score_rag"] >= example["score_full"] else "full"
    opposite_strategy = "full" if better_strategy == "rag" else "rag"
    is_correct = True if better_strategy in routes else False
    return {
        "example_id": example["_id"],
        "decision_prompt": decision_prompt,
        "example": example,
        "route_decision": better_strategy if is_correct else opposite_strategy,
        "better_strategy": better_strategy,
        "is_correct": is_correct,
        "reason": reason,
        "routes": routes,  # 保存所有路由决策
        "thinking_trace": thinkings,  # 保存所有思考过程
        "model": model,
        "doc_type": doc_type,
        "doc_len_tokens": doc_len_tokens,
        "doc_head": context_snippet[:head_n],
        "ground_truth_option": example.get("ground_truth_option"),
        "rag_raw_output": example.get("rag_raw_output"),
        "rag_final_choice": example.get("rag_final_choice"),
        "lc_raw_output": example.get("lc_raw_output"),
        "lc_final_choice": example.get("lc_final_choice"),
        "rag_correct": example.get("rag_correct"),
        "lc_correct": example.get("lc_correct")
    }

def sample_json(input_path, output_path, sample_size=30, seed=42):
    if os.path.exists(output_path):
        print(f"采样文件已存在，直接复用: {output_path}")
        return output_path
    random.seed(seed)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) <= sample_size:
        print(f"数据量小于采样数 {sample_size}，全部保留")
        sampled = data
    else:
        sampled = random.sample(data, sample_size)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"采样完成，采样数: {len(sampled)}，保存到: {output_path}")
    return output_path


def sample_json_by_id(input_path, output_path, sample_size=30, seed=42, global_ids=None):
    """基于_id进行采样以确保一致性"""
    if os.path.exists(output_path):
        print(f"采样文件已存在，直接复用: {output_path}")
        return output_path
    
    random.seed(seed)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 为数据添加_id
    data = add_ids_to_data(data)
    
    if len(data) <= sample_size:
        print(f"数据量小于采样数 {sample_size}，全部保留")
        sampled = data
    else:
        if global_ids is None:
            sampled = random.sample(data, sample_size)
        else:
            # 使用全局ID进行采样以确保一致性
            id_to_example = {example["_id"]: example for example in data}
            sampled = [id_to_example[id_] for id_ in global_ids if id_ in id_to_example]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"采样完成，采样数: {len(sampled)}，保存到: {output_path}")
    return output_path


def main(
    input_json: str,
    router_model: str,
    max_workers: int = 4,
    best_of_n: int = 1,
    temperature: float = 0.0,
    prompt_version: str = "v2",
    head_n: int = 2000,
    rag_chunk_mapping: dict = {},
    thinking: bool = False,
    rewrite: bool = False,
    answer_model: str = "qwen3-235b-a22b-instruct-2507",
    test_ids: Optional[set] = None,
):
    """
    Router judge 评测主函数（支持 test_ids ）：
    - 若输出已存在且不重写：直接加载结果；若提供 test_ids，先按 test_ids 过滤，再算指标。
    - 若输出不存在或 rewrite=True：
        * 读取输入数据；若提供 test_ids，则仅对 test 子集推理；
        * 推理完成后保存并计算指标。
    """
    # 使用统一的路径生成函数
    output_file = _router_judge_output_path(
        input_json=input_json,
        router_model=router_model,
        prompt_version=prompt_version,
        best_of_n=best_of_n,
        router_thinking=thinking
    )

    # ---- 已有结果直接复用 ----
    if os.path.exists(output_file) and not rewrite:
        print(f"输出文件已存在，直接复用: {output_file}")
        json_data = load_json(output_file)

        # 若指定 test_ids，只对 test 子集计算指标
        if test_ids:
            json_data = filter_examples_by_ids(json_data, test_ids)
            if not json_data:
                print("[split indices] 警告：现有输出与 test_ids 无交集，无法计算指标。")
                return None, None, None
        # 统计type X level分布
        type_level_stats = count_level_type_ctx_distribution(json_data)
        print_ctx_type_level_distribution(type_level_stats)
        # 归并 long_context -> full
        for item in json_data:
            if item.get("route_decision") in ["long_context", "full"]:
                item["route_decision"] = "full"

        json_data = expand_example_field(json_data)
        def check_route_decision(item):
            if "routes" not in item:
                return item
            # 根据第219行的逻辑重新设置route_decision
            better_strategy = "rag" if item["score_rag"] >= item["score_full"] else "full"
            opposite_strategy = "full" if better_strategy == "rag" else "rag"
            is_correct = True if better_strategy in item.get("routes") else False
            
            item["route_decision"] = better_strategy if is_correct else opposite_strategy
            return item
        json_data = [check_route_decision(item) for item in json_data]
        strict_metrics, lenient_metrics, equal_included_metrics = compute_and_print_metrics(json_data)
        return strict_metrics, lenient_metrics, equal_included_metrics
    
    if os.path.exists(output_file) and rewrite:

        backup_file = output_file.replace(".json", f"_backup_{time.strftime('%Y%m%d%H%M%S')}.json")
        shutil.copy(output_file, backup_file)
        print(f"备份已建立：{backup_file}")
        

    data = load_json(input_json)
    print(f"加载 {len(data)} 个项目用于路由评估")

    # 若提供 test_ids，则只对 test 子集推理
    if test_ids:
        data = filter_examples_by_ids(data, test_ids)
        print(f"[split indices] 仅对 test_ids 子集推理：{len(data)} 个样本")
        if not data:
            print("[split indices] 警告：输入数据与 test_ids 无交集。")
            return None, None, None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_item,
                example,
                router_model,
                answer_model,
                best_of_n,
                temperature,
                prompt_version,
                head_n,
                rag_chunk_mapping,
                thinking
            ): example for example in data
        }
        for future in tqdm(as_completed(futures), total=len(data), desc="Processing items", mininterval=20):
            result = future.result()
            if result:
                results.append(result)

    save_json(output_file, results)
    print(f"处理完成，结果已保存到 {output_file}")

    if results:
        # 归并 long_context -> full
        for item in results:
            if item.get("route_decision") in ["long_context", "full"]:
                item["route_decision"] = "full"

        json_data = expand_example_field(results)
        strict_metrics, lenient_metrics, equal_included_metrics = compute_and_print_metrics(json_data)
        return strict_metrics, lenient_metrics, equal_included_metrics

    else:
        print("没有足够的数据来计算评估指标")
        return None, None, None


# ====== 直接粘贴到你的脚本中（例如放在 import 之后、main 之前）======

from collections import defaultdict

def _get_input_jsons(answer_model: str, compare_model: str) -> list:
    """返回该 (answer_model, compare_model) 下实际存在的输入文件列表。"""
    thinking_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_thinking_comparison.json"
    instruct_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_comparison.json"
    input_jsons = []
    if os.path.exists(thinking_path):
        input_jsons.append(thinking_path)
    if os.path.exists(instruct_path):
        input_jsons.append(instruct_path)
    return input_jsons


def _router_judge_output_path(input_json: str, router_model: str, prompt_version: str, best_of_n: int, router_thinking: bool) -> str:
    """按照你现有 main 的规则生成输出文件路径。"""
    return input_json.replace(
        ".json",
        f"_router{router_model}_{prompt_version}_bon{best_of_n}_thinking{router_thinking}.json"
    )


def enumerate_all_tasks(args, update_rag: bool = True) -> list:
    """
    枚举所有 router_judge 任务，
    每个任务包含其输出文件路径，供统计/去重/继续执行使用。
    返回一个任务字典列表，每项格式：
    {
      "type": "router_judge",
      "answer_model": ...,
      "compare_model": ...,
      "router_model": ...,
      "input_json": ...,
      "router_thinking": bool,
      "prompt_version": str | None,
      "out_path": str
    }
    """
    tasks = []
    for best_of_n in args.best_of_ns:
        for answer_model in args.answer_models:
            for compare_model in args.compare_models:

                input_jsons = _get_input_jsons(answer_model, compare_model)
                if not input_jsons:
                    continue
                for router_model in args.router_models:
                    # 由你现有函数决定 router_thinking 的取值（如 [False] 或 [False, True]）
                    thinkings = use_thinking_for_model(router_model) if 'use_thinking_for_model' in globals() else [False]
                    for input_json in input_jsons:
                        for router_thinking in thinkings:
                            # 2) main (router_judge) 任务：遍历 prompt_versions
                            for prompt_version in args.prompt_versions:
                                out_main = _router_judge_output_path(
                                    input_json, router_model, prompt_version, best_of_n, router_thinking
                                )
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
    """
    汇总统计：总任务数 / 已完成 / 未完成。
    - show_missing_examples: 若 >0，将额外打印若干条“未完成任务”的示例，便于确认。
    返回一个 summary 字典，便于程序内进一步使用。
    """
    tasks = enumerate_all_tasks(args, update_rag=update_rag)
    total = len(tasks)
    done = sum(1 for t in tasks if os.path.exists(t["out_path"]))
    remaining = total - done

    # 分类统计
    by_type = defaultdict(lambda: {"total": 0, "done": 0})
    for t in tasks:
        by_type[t["type"]]["total"] += 1
        if os.path.exists(t["out_path"]):
            by_type[t["type"]]["done"] += 1

    print("\n========== 任务进度统计 ==========")
    print(f"总任务数: {total}")
    print(f"已完成 : {done}")
    print(f"未完成 : {remaining}")

    # 各类型明细
    for k, v in by_type.items():
        print(f"- {k}: {v['done']}/{v['total']} 已完成")

    # 可选：给出若干条未完成样例
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

    # 结构化返回
    summary = {
        "total": total,
        "done": done,
        "remaining": remaining,
        "by_type": {k: {"total": v["total"], "done": v["done"], "remaining": v["total"] - v["done"]} for k, v in by_type.items()},
        "tasks": tasks,  # 如需后续程序使用可保留；纯统计也可去掉
    }
    print("=================================\n")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="路由评估脚本+采样缓存")
    parser.add_argument("--answer_models", nargs='+', default=[
        # "qwen3-235b-a22b-instruct-2507",
        # "qwen3-235b-a22b-thinking-2507",
        # "deepseek-r1-0528",
        # "qwen-max-latest",
        # "qwen3-1.7b",
        # "qwen3-4b",
        # "qwen3-8b",
        "qwen3-30b-a3b-instruct-2507",
        # "qwen3-30b-a3b-thinking-2507",
    ], help="回答模型列表")
    parser.add_argument("--router_models", nargs='+', default=[
        # "local_sft_qwen3_1_7b",
        # "qwen3-1.7b",
        # "qwen3-4b",
        # "qwen3-8b",
        # "qwen3-30b-a3b-instruct-2507",
        # "qwen3-30b-a3b-thinking-2507",
        "qwen3-235b-a22b-instruct-2507",
        # "qwen3-235b-a22b-thinking-2507",
        # "deepseek-r1-0528",
        # "qwen-max-latest"
    ], help="路由模型列表")
    parser.add_argument("--compare_models", nargs='+', default=[
        # "qwen-max", 
        "qwen3-235b-a22b-instruct-2507"
    ], help="对比模型列表")


    parser.add_argument("--sample", default=False, help="是否只用采样后的文件评测")
    parser.add_argument("--sample_size", type=int, default=200, help="采样数量")
    parser.add_argument("--sample_seed", type=int, default=42, help="采样随机种子")
    parser.add_argument(
    "--best_of_ns", nargs='+', type=int, default=[1],
    help="生成时的 best_of_n 列表，如: --best_of_ns 1 3 5"
)
    parser.add_argument("--prompt_versions", nargs='+', default=["preroute",], type=str,
                        choices=["preroute"],
                        help="prompt type (preroute = full Pre-Route guidelines)")
    parser.add_argument('--ragchunk_pkl', type=str, default=None, help='pkl file with rag chunk cache')
    parser.add_argument("--head_n", type=int, default=2000, help="rag chunk cache head n")
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default="distillation_datasets/from_multi_answer_models_0.7_0.1_0.2_42/global_split_indices.json",
        help="可选：global_split_indices.json 的路径；若提供，将使用其中的 test_ids 进行筛选。"
    )
    parser.add_argument("--rewrite", help="是否重写", default=True)
    # parser.add_argument("--thinking", action="store_true", help="是否使用思考版本", default=False)
    args = parser.parse_args()

    # 读取 test_ids（若提供）
    test_ids = load_test_ids(args.split_indices_path)

    # 若存在 test_ids，则忽略随机采样逻辑，严格用 test 集
    if test_ids:
        print(f"[split indices] 已加载 test_ids 数量：{len(test_ids)}，将忽略随机采样，严格按 test_ids 评测。")
    else:
        print("[split indices] 未启用或加载失败，保持现有评测/采样流程。")


    rewrite = args.rewrite

    ragchunk_pkl = args.ragchunk_pkl or f"./datasets/all_query_map_ragchunks.pkl"

    if os.path.exists(ragchunk_pkl):
        print(f"加载RAG缓存: {ragchunk_pkl}")
        with open(ragchunk_pkl, "rb") as f:
            rag_chunk_mapping = pickle.load(f)
        print(f"RAG缓存条目数: {len(rag_chunk_mapping)}")
    else:
        print(f"未找到RAG缓存 {ragchunk_pkl}，后续全部实时retrieval！")
        rag_chunk_mapping = {}


    # 生成 input_jsons 列表
    all_input_jsons = []
    for compare_model in args.compare_models:
        input_jsons = []
        for answer_model in args.answer_models:
            # 检查 thinking 版本
            thinking_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_thinking_comparison.json"
            if os.path.exists(thinking_path):
                input_jsons.append(thinking_path)
            
            # 检查非 thinking 版本
            instruct_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_comparison.json"
            if os.path.exists(instruct_path):
                input_jsons.append(instruct_path)



        # print(f"对比模型 {compare_model} : 找到 {len(input_jsons)} 个有效的输入文件")
        print(input_jsons)
        all_input_jsons.extend(input_jsons)
    print(f"总共有 {len(all_input_jsons)} 个有效的输入文件")
    print(all_input_jsons)
    ## add id to all input jsons
    for input_json in all_input_jsons:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "_id" not in data[0]:
            data = add_ids_to_data(data)
            with open(input_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    # 采样及缓存逻辑


    sample_jsons = []
    if test_ids:
        # 不做随机采样，直接使用 all_input_jsons
        sample_jsons = all_input_jsons
    else:
        if args.sample:
            # 生成全局ID
            first_file_data = []
            if all_input_jsons:
                with open(all_input_jsons[0], 'r', encoding='utf-8') as f:
                    first_file_data = json.load(f)
            
            # 生成全局ID文件路径
            global_ids_file = f"./datasets/global_ids_sample{args.sample_size}_seed{args.sample_seed}.json"
            
            # 如果全局ID文件存在，则加载它
            if os.path.exists(global_ids_file):
                print(f"加载已存在的全局ID文件: {global_ids_file}")
                with open(global_ids_file, 'r', encoding='utf-8') as f:
                    global_ids = json.load(f)
            else:
                # 否则生成新的全局ID并保存
                random.seed(args.sample_seed)
                global_ids = [example["_id"] for example in random.sample(first_file_data, min(args.sample_size, len(first_file_data)))]
                with open(global_ids_file, 'w', encoding='utf-8') as f:
                    json.dump(global_ids, f, indent=2, ensure_ascii=False)
                print(f"生成并保存全局ID到: {global_ids_file}")
            
            for input_json in all_input_jsons:
                out_dir = os.path.dirname(input_json)
                base_name = os.path.basename(input_json).replace(".json", f"_sample{args.sample_size}.json")
                out_path = os.path.join(out_dir, base_name)
                sample_json_by_id(input_json, out_path, args.sample_size, args.sample_seed, global_ids)
                sample_jsons.append(out_path)
        else:
            sample_jsons = all_input_jsons

    _ = summarize_task_progress(args, update_rag=True, show_missing_examples=5)
    # ======================= 保存评估结果 =======================
    res_metrics = []
    # 主评测流程
    for best_of_n in args.best_of_ns:
        print("==ARGS==")
        print(f"answer_models: {args.answer_models}")
        print(f"router_models: {args.router_models}")
        print(f"compare_models: {args.compare_models}")
        print(f"sample: {args.sample}")
        print(f"sample_size: {args.sample_size}")
        print(f"sample_seed: {args.sample_seed}")
        print(f"best_of_n: {best_of_n}")
        print(f"prompt_versions: {args.prompt_versions}")
        print(f"ragchunk_pkl: {args.ragchunk_pkl}")
        print(f"head_n: {args.head_n}")

        # 扁平化记录列表：每条记录就是一个配置的结果（更易解析）


        if best_of_n <= 0:
            temperature = 0
            best_of_n_eff = 1  # 注意：只改变内部使用的变量，不回写到 args 或循环变量
            print(f"Using Greedy Search, setting temperature to {temperature}")
        else:
            temperature = 1
            best_of_n_eff = best_of_n
            print(f"Using BoN= {best_of_n_eff}, setting temperature to {temperature}")

        for answer_model in args.answer_models:


            for compare_model in args.compare_models:
                
                input_jsons = []
                thinking_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_thinking_comparison.json"
                if os.path.exists(thinking_path):
                    input_jsons.append(thinking_path)
                
                # 检查answer model非 thinking 版本
                instruct_path = f"evaluation/prediction/{answer_model}/flattened_merged_predictions_{compare_model}_comparison.json"
                if os.path.exists(instruct_path):
                    input_jsons.append(instruct_path)
                for router_model in args.router_models:
                    # 检查answer model thinking 版本
                    
                    thinkings = use_thinking_for_model(router_model)

                    # 遍历每个输入文件
                    for input_json in input_jsons:
                        for router_thinking in thinkings:
                            max_workers = int(get_max_workers_for_model(router_model))
                            answer_thinking_flag = "thinking_comparison.json" in input_json

                            # ===== router_judge：对每个 prompt_version 跑一次（保持原逻辑）=====
                            for prompt_version in args.prompt_versions:
                                print(f"====router judge:====")
                                print(f"max_workers: {max_workers}")
                                print(f"prompt_version: {prompt_version}")
                                print(f"input json: {input_json}, compare model: {compare_model}, answer model: {answer_model} router_model: {router_model}, thinking: {router_thinking}")
                                # rewrite = True
                                strict_metrics, lenient_metrics, equal_included_metrics = main(
                                    input_json=input_json,
                                    router_model=router_model,
                                    max_workers=max_workers,
                                    best_of_n=best_of_n_eff,
                                    temperature=temperature,
                                    prompt_version=prompt_version,
                                    head_n=args.head_n,
                                    rag_chunk_mapping=rag_chunk_mapping,
                                    thinking=router_thinking,
                                    rewrite=rewrite,
                                    test_ids=test_ids,
                                    answer_model=answer_model,
                                )
                                # 扁平记录一条 router_judge
                                res_metrics.append({
                                    "answer_model": answer_model,
                                    "router_model": router_model,
                                    "compare_model": compare_model,
                                    "answer_thinking": bool(answer_thinking_flag),
                                    "router_thinking": bool(router_thinking),
                                    "kind": "router_judge",
                                    "prompt_version": prompt_version,
                                    "best_of_n": int(best_of_n_eff),
                                    "strict": strict_metrics,
                                    "lenient": lenient_metrics,
                                    "equal_included": equal_included_metrics,
                                })
                                

    # ===== 保存为 JSON（文件名包含整组 best_of_ns，保持你原有命名习惯）=====
    out_json_path = f"evaluation/prediction/lara_router_metrics_{bool(args.split_indices_path)}_{args.best_of_ns}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(res_metrics, f, indent=2, ensure_ascii=False)
    print(f"saving to {out_json_path}")

    # ===== 生成 Markdown 总表（基于扁平 res_metrics 即时渲染）=====
    def _pick_block(md: dict):
        for key in [ "lenient", "equal_included", "strict"]:
            if isinstance(md.get(key), dict):
                return md[key], key
        return None, None

    def _fmt_score(v):
        if v is None:
            return "-"
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "-"

    def _fmt_pct(num, den):
        if not num or not den:
            return "-"
        try:
            return f"{(num/den)*100:.1f}%"
        except Exception:
            return "-"

    def _fmt_bool(b):
        return "✓" if bool(b) else "✗"

    # 生成一行汇总
    lines = []
    lines.append("## 总表\n")
    lines.append("| **Answer Model** | **Answer Thinking** | **Kind** | **Router Model** | **Router Thinking** | **Prompt Ver** | **BoN** | **最终问答得分 (效果)** | **LC选择率 (成本)** | **Acc** |")
    lines.append("| ---------------- | ------------------- | -------- | ---------------- | ------------------- | ------------- | ------ | ----------------------- | ------------------- | ------------------------------ |")

    # 排序：Answer -> Router -> PV -> BoN
    def _sort_key(rec):
        return (
            rec["answer_model"],
            rec["router_model"],
            str(rec["prompt_version"]),
            rec["best_of_n"],
            bool(rec["router_thinking"]),
            bool(rec["answer_thinking"]),
        )

    for rec in sorted(res_metrics, key=_sort_key):
        # 取 block
        block, _ = _pick_block(rec)
        if not block:
            # skip if no metrics block
            continue

        score   = block.get("actual_score_avg")
        lc_rate = _fmt_pct(block.get("lc_count"), block.get("total_examples"))
        acc     = block.get("accuracy")
        auc     = block.get("auc")
        acc_auc = f"{_fmt_score(acc)}" if (acc is not None) else "-"

        lines.append(
            f"| {rec['answer_model']} | {_fmt_bool(rec['answer_thinking'])} | {rec['kind']} | "
            f"{rec['router_model']} | {_fmt_bool(rec['router_thinking'])} | {rec['prompt_version']} | {rec['best_of_n']} | "
            f"{_fmt_score(score)} | {lc_rate} | {acc_auc} |"
        )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    md_text = "\n".join(lines)
    print(md_text)

    # 另存一份 md
    out_md_path = out_json_path.replace(".json", ".md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"markdown saved to {out_md_path}")
    # ======================= 保存评估结果 =======================