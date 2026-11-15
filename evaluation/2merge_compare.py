import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval_utils import compare_two_answers
from tqdm import tqdm

TEMP_SAVE_FILE = "temp_comparison_results.json"

def merge_jsonl_files(data_dir, output_file, thinking=False):
    merged = {}
    for fname in os.listdir(data_dir):
        if not (fname.endswith(".jsonl") and ((thinking and "thinking" in fname) or ((not thinking) and "thinking" not in fname))):
            continue
        parts = fname.replace(".jsonl", "").split("_")
        if len(parts) < 6:
            print(f"[⚠️ 跳过] 非标准命名: {fname}")
            continue
        strategy = parts[0]
        context_length = parts[3]
        model = parts[2]
        context_type = parts[4]
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[❌ 错误] JSON 无法解析: {line[:80]}...")
                    continue
                key = (item["file"], item["question"])
                if key not in merged:
                    merged[key] = {
                        "file": item["file"],
                        "question": item["question"],
                        "ground_truth": item.get("ground_truth", ""),
                        "type": item.get("type", ""),
                        "level": item.get("level", ""),
                        "comp_parts": item.get("comp_parts", []),
                        "context_length": context_length,
                        "context_type": context_type,
                    }
                if isinstance(item.get("prediction"), list):
                    item["prediction"] = item["prediction"][1]
                merged[key][strategy] = item.get("prediction", "")
    final_data = list(merged.values())
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 扁平合并完成，输出路径: {output_file}，共计 {len(final_data)} 条记录")
    return output_file

def process_item(item, judge_model: str, answer_model: str):
    if "rag" not in item or "full" not in item:
        return None
    if item["rag"] == False or item["full"] == False:
        return None

    if isinstance(item["rag"], str):
        pred_rag = item.get("rag", "").strip()
        pred_full = item.get("full", "").strip()
    elif isinstance(item["rag"], list):
        pred_rag = item["rag"][1].get("prediction", "").strip()
        pred_full = item["full"][1].get("prediction", "").strip()
    else:
        raise ValueError(f"Unknown type for rag or full: {type(item['rag'])}")
    groundtruth = item.get("ground_truth", "").strip()
    question = item.get("question", "")
    context_type = item.get("context_type", "")
    file_name = item.get("file", "")
    if not pred_rag or not pred_full:
        item["comparison_analysis"] = "Missing one or both predictions."
        item["score_rag"] = 1
        item["score_full"] = 1
        item["better_model"] = "Tie"
        return item
    try:
        result = compare_two_answers(
            pred_a=pred_rag,
            pred_b=pred_full,
            label=groundtruth,
            query=question,
            query_type=context_type, 
            model=judge_model
        )
        item["comparison_analysis"] = result["analysis"]
        item["score_rag"] = result["score_a"]
        item["score_full"] = result["score_b"]
        item["better_model"] = (
            "RAG" if result["better"] == "A"
            else "FULL" if result["better"] == "B"
            else "Tie"
        )
    except Exception as e:
        print(f"[ERROR] Failed to compare item (file={file_name}, question={question}): {e}")
        item["comparison_analysis"] = "Evaluation failed."
        item["score_rag"] = 1
        item["score_full"] = 1
        item["better_model"] = "Tie"
    item["model"] = answer_model
    return item

def save_temp_results(updated_data, counter):
    with open(TEMP_SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "counter": counter,
            "updated_data": updated_data
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Temporary results saved after {counter} items")

def load_temp_results():
    if os.path.exists(TEMP_SAVE_FILE):
        with open(TEMP_SAVE_FILE, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
        return temp_data["counter"], temp_data["updated_data"]
    return 0, []

def compare_main(json_path: str, max_workers: int, judge_model: str = "qwen-max", thinking: bool = False):
    output_file = json_path.replace(".json", f"_{judge_model}{'_thinking' if thinking else ''}_comparison.json")

    if os.path.exists(output_file):
        print(f"✅ 输出文件已存在: {output_file}")
        return

    answer_model = os.path.basename(os.path.dirname(json_path))  # 更鲁棒

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) == 0:
        print(f"⚠️ 输入文件 {json_path} 为空")
        return
    # print(f"🔍 Loaded {len(data)} items from {json_path}")
    # start_counter, updated_data = load_temp_results()
    start_counter = 0
    updated_data = []
    if start_counter > 0:
        print(f"🔄 Resuming from {start_counter} completed items")
        processed_keys = {(item.get('file'), item.get('question')) for item in updated_data}
        data = [
            item for item in data
            if (item.get('file'), item.get('question')) not in processed_keys
            and item.get("rag") not in [False, None, ""]
            and item.get("full") not in [False, None, ""]
        ]
        print(f"Remaining items to process: {len(data)}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item, judge_model, answer_model): item for item in data}
        temp_save_counter = start_counter
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Comparing items...")):
            updated_item = future.result()
            if updated_item is None:
                continue
            updated_data.append(updated_item)
            temp_save_counter += 1
            if temp_save_counter % 100 == 0:
                save_temp_results(updated_data, temp_save_counter)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    print(f"🎉 所有比较完成，结果已保存至: {output_file}")
    if os.path.exists(TEMP_SAVE_FILE):
        os.remove(TEMP_SAVE_FILE)
        print(f"🧹 Temporary file {TEMP_SAVE_FILE} removed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step1: merge jsonl; Step2: compare merged.")
    parser.add_argument("--data_dir", type=str, default="evaluation/prediction/qwen3-30b-a3b-instruct-2507", help="Directory containing jsonl files.")
    parser.add_argument("--workers", type=int, default=30, help="Number of threads for parallel comparison.")
    parser.add_argument("--judge_model", type=str, default="qwen3-235b-a22b-instruct-2507", help="Judge model to use for comparison.")
    parser.add_argument("--thinking", action="store_true", default=False, help="Whether to compare thinking results.")

    args = parser.parse_args()
    merged_json_path = os.path.join(args.data_dir, "flattened_merged_predictions.json")



    print("正在合并所有jsonl为扁平json...")
    merge_jsonl_files(args.data_dir, merged_json_path, thinking=args.thinking)


    # step2: compare
    compare_main(merged_json_path, max_workers=args.workers, judge_model=args.judge_model, thinking=args.thinking)

    os.remove(merged_json_path)
    print(f"🧹 合并文件 {merged_json_path} 已移除")