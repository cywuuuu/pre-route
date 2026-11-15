from openai import OpenAI
import tiktoken
from pathlib import Path
from eval_utils import (
    load_jsonl,
    create_msgs,
    dump_jsonl,
    call_model,
    get_tokenizer
)
import os
import time
import json
import requests
import concurrent.futures
import dashscope
import time
import json
import os
import traceback
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query_type', type=str, default="hallu", help='the type of the query')
parser.add_argument('--context_type', type=str, default="financial", help='the type of the context')
parser.add_argument('--context_length', type=str, default="32k", help='the length of the context')
parser.add_argument('--eval_model', type=str, default="qwen3-30b-a3b-instruct-2507", help='model')
parser.add_argument('--thinking', action='store_true', default=False)
parser.add_argument('--rewrite', action='store_true', default=False)
parser.add_argument('--limit', type=int, default=None, help='process only the first N queries (low-cost trial)')
'''
QUERY_TYPES=("hallu" "location" "reasoning" "comp")
CONTEXT_TYPES=("financial" "paper" "book")
CONTEXT_LENGTHS=("32k" "128k")
'''
args = parser.parse_args()
eval_model = args.eval_model
query_type = args.query_type
context_type = args.context_type
context_length = args.context_length


def process_example(eg, args):
    file_name = eg['file']
    eg_file_path = f"./datasets/{context_length}/{context_type}/{file_name}"
    with open(eg_file_path, 'r', encoding='utf-8') as f:
        eg_txt = f.read()
    eg['context'] = eg_txt 

    reserved_ratio_tries = [0.9, 0.8, 0.7, 0.6]
    for reserved_ratio in reserved_ratio_tries:
        try:
            msgs, prompt = create_msgs(
                tokenizer, eg, data_name=context_type, model_name="full", 
                eval_model=eval_model, reserved_ratio=reserved_ratio
            )
            response = call_model(msgs, eval_model, thinking=args.thinking)

            # print("response:", response)
            # 判断是否超长报错
            if isinstance(response, str) and (
                ("input length should" in response)
            ):
                print(f"[⚠️] reserved_ratio={reserved_ratio} 仍然过长，尝试更小 reserved_ratio")
                continue
            return response, eg
        except Exception as e:
            print(f"ERROR with reserved_ratio={reserved_ratio}: {e}")
            print("ERR FULL")
            print(context_length, context_type, tokenizer, eval_model)
            continue
        time.sleep(2)
    # 全部失败
    print(f"[❌] All reserved_ratio attempts failed for {file_name}")
    return "ERROR: input too long after all truncations", eg
if __name__ == "__main__":

    data_path = f'./datasets/query/{context_length}_{context_type}_{query_type}.jsonl'
    examples = load_jsonl(data_path)
    if args.limit:
        examples = examples[:args.limit]
        print(f"limit: processing first {len(examples)} queries")
    output_dir = f"./evaluation/prediction/{eval_model}"
    output_path = f'{output_dir}/full_preds_{eval_model}_{context_length}_{context_type}_{query_type}{"_thinking" if args.thinking else ""}.jsonl'
    print(f"output_path: {output_path}")
    if os.path.exists(output_path) and not args.rewrite:
        print(f"已存在预测结果 {output_path}，跳过！")
        exit(0)
    os.makedirs(output_dir, exist_ok=True)
    print(f"make dirs: {output_dir}")
    # tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokenizer = get_tokenizer(eval_model)
    preds = []
    print(f"===processing：===\n")
    print(f"eval_model: {args.eval_model}")
    print(f"context_length: {args.context_length}")
    print(f"context_type: {args.context_type}")
    print(f"query_type: {args.query_type}")
    print(f"thinking: {args.thinking}")
    print(f"rewrite: {args.rewrite}")

    eval_model = args.eval_model
    if eval_model in ['qwen3-235b-a22b-thinking-2507', 'qwen3-235b-a22b-instruct-2507']:
        max_workers = 10
    elif eval_model in ['deepseek-r1-0528', "qwen-max-latest"]:
        max_workers = 4
    else:
    # 保持原有的默认逻辑
        if 'qwen' in eval_model:
            max_workers = 16
        else:
            max_workers = 10
    if context_length == "32k":
        max_workers *= 1.5
        max_workers = int(max_workers)
    print(f"true max_workers: {max_workers}")
    # for i in range(10):
    #     response, eg = process_example(examples[i])
    #     breakpoint()
    from tqdm import tqdm
    # if not os.path.exists(output_path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_example, example, args) for example in examples]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing examples"):
            result, eg = future.result()
            
            if query_type in ['location', 'reasoning', 'hallu']:
                preds.append({
                    "type": eg['type'],
                    "level": eg['level'],
                    "file": eg['file'],
                    "context_order": eg['context_order'],
                    "question": eg['question'],
                    "reason": result[0],
                    "prediction": result[1],
                    "ground_truth": eg['answer'],
                })
            
            elif query_type in ['comp']:
                preds.append({
                    "type": eg['type'],
                    "level": eg['level'],
                    "file": eg['file'],
                    "comp_parts": eg['comp_parts'],
                    "question": eg['question'],
                    "reason": result[0],
                    "prediction": result[1],
                    "ground_truth": eg['answer'],
                })

    dump_jsonl(preds, output_path)
    print(f"rag_preds_{eval_model}_{context_length}_{context_type}_{query_type} saving to {output_path}")