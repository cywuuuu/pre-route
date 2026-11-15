import json
import re
from typing import Optional
import math
import os
from prompt import (
    full_templates,
    rag_templates
)
import tiktoken
import hashlib
from transformers import AutoTokenizer
import tiktoken
from typing import List, Dict, Optional, Tuple
enc = tiktoken.encoding_for_model("gpt-4")
MODEL_TO_PROMPT_TEMPLATE = {
    "full": full_templates,
    "rag": rag_templates
}
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import sentencepiece as spm

def get_tokenizer(eval_model: str):
    name = eval_model.lower()
    # Qwen系列
    if "qwen" in name.lower():
        return AutoTokenizer.from_pretrained("model_weights/Qwen3-1.7B")  # 或最新的Qwen3模型
    elif "gemma" in name or "gemini" in name:
        return AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        return sp
    elif name.startswith("gpt-3") or name.startswith("gpt-4"):
        return tiktoken.encoding_for_model("gpt-4")
    elif "deepseek" in name:
        return AutoTokenizer.from_pretrained("model_weights/DeepSeek-R1-tokenizer")
    else:
        return tiktoken.encoding_for_model("gpt-4")

from openai import OpenAI
import dashscope
import time
local_sft_model = ("local", "sft", "model_weights") 
aliapi_model = ("qwen", "deepseek", "qwq", "llama", "qvq", "plus", "Kimi", "glm")
oprouter_model = ("claude")
gpt_model = ("gpt")
google_model = ("gemini")
# 定义模型匹配关键词与对应API Key的映射
MODEL_API_KEY_MAPPING = {
    local_sft_model: "sk-local-placeholder",
    "zgc": os.environ.get("PREROUTE_ZGC_API_KEY", ""),
    aliapi_model: os.environ.get("DASHSCOPE_API_KEY", ""),
    oprouter_model: os.environ.get("OPENROUTER_API_KEY", ""),
    gpt_model: os.environ.get("OPENAI_API_KEY", ""),
    google_model: os.environ.get("GOOGLE_API_KEY", ""),
}

# 默认API Key
DEFAULT_API_KEY = os.environ.get("PREROUTE_DEFAULT_API_KEY", "")

# 定义模型匹配关键词与对应URL的映射
MODEL_URL_MAPPING = {
    local_sft_model: "http://127.0.0.1:8000/v1",
    aliapi_model: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    oprouter_model: "https://openrouter.ai/api/v1",
    gpt_model: "https://api.openai.com/v1/chat/completions",
    google_model: "https://generativelanguage.googleapis.com/v1beta/openai/",

}

def get_api_key(eval_model):
    for key, api_key in MODEL_API_KEY_MAPPING.items():
        if isinstance(key, tuple):
            if any(k in eval_model for k in key):
                return api_key
        elif key in eval_model:
            return api_key
    return DEFAULT_API_KEY

def get_url(eval_model):
    for key, url in MODEL_URL_MAPPING.items():
        if isinstance(key, tuple):
            if any(k in eval_model for k in key):
                return url
        elif key in eval_model:
            return url
    return None

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
def load_jsonl(data_path):
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list

def load_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def truncate_input(input, max_length, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "sc":
        num_parts = math.ceil(len(input) / max_length)
        all_parts = []
        for i in range(num_parts):
            if i < (num_parts - 1):
                all_parts.append(input[i * max_length : (i + 1) * max_length])
            else:
                all_parts.append(input[i * max_length :])
        return all_parts
    if manner == "middle":
        return input[0 : max_length // 2] + input[-max_length // 2 :]
    else:
        return None

def get_max_workers_for_model(eval_model):

    if eval_model in ['qwen3-235b-a22b-thinking-2507', 'qwen3-235b-a22b-instruct-2507']:
        return 24
    elif eval_model in ["qwen-max-latest"]:
        return 8
    elif eval_model in ['deepseek-r1-0528' ]:
        return 24
    elif eval_model in ["glm-4.5", "llama-4-maverick-17b-128e-instruct", "Moonshot-Kimi-K2-Instruct",]:
        return 14
    else:
        # 保持原有的默认逻辑
        if "sft" in eval_model:
            return 8
        if 'qwen' in eval_model:
            return 30
        else:
            return 8

def generate_id(example: Dict) -> str:
    """为样本生成唯一标识符 (基于问题本身)"""
    # 确保所有必要的键都存在，避免KeyError
    keys_to_hash = ["file", "question", "type", "level"]
    if not all(key in example for key in keys_to_hash):
        # 如果缺少关键信息，则使用一个更通用的哈希，但这应该很少见
        return hashlib.md5(json.dumps(example, sort_keys=True).encode()).hexdigest()
    
    return hashlib.md5((
        example.get("file", "") + 
        example.get("question", "") 
    ).encode()).hexdigest()

def create_msgs(tokenizer, eg: dict, data_name: str, model_name:str, eval_model:str, thinking=False, reserved_ratio=0.95) -> tuple[list[dict], str]:
    """
    Create messages for a given example.
    """
    prompt = create_prompt(eg, data_name, model_name)
    max_token_num = get_max_token_length(eval_model, thinking=thinking)  
    # Check if tokenizer is provided and initialized
    if tokenizer and model_name == 'full':
        tokens = tokenizer.encode(prompt)
        print(f"Before truncation: {len(tokens)}")
        tokens = truncate_input(tokens, int(max_token_num * reserved_ratio), manner="middle")
        print(f"After truncation: {len(tokens)}")  # type: ignore
        prompt = tokenizer.decode(tokens)

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant",  # noqa
        },  # noqa
        {"role": "user", "content": prompt},
    ], prompt

def create_prompt(eg: dict, data_name: str, model_name: Optional[str]) -> str:
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
        model_name: optional, used to fetch model-specific templates.
    """

    # Directly use the appropriate template if the model_name is provided.
    if model_name and model_name in MODEL_TO_PROMPT_TEMPLATE:
        templates = MODEL_TO_PROMPT_TEMPLATE[model_name]
        template = templates[data_name]
    else:
        # If no model-specific template, return a basic prompt or handle differently.
        return eg["context"]

    # Now create the prompt based on the template and task data

    if data_name in [
        "longbook_choice_eng",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longbook_qa_chn",
    ]:
        book = eg["context"]
        if data_name == "longbook_choice_eng":
            return template.format(
                question=eg["input"],
                context=book,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        elif data_name == "longbook_qa_eng":
            return template.format(
                question=eg["question"],
                context=book,
            )
        else:
            raise ValueError

    format_dict = {
        "context": eg["context"],
        "question": eg["question"],
    }
    prompt = template.format(**format_dict)
    return prompt

def dump_jsonl(data, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

import traceback

# ========== 模型调用 ==========
def call_gpt(model, messages, retry_num=5, retry_interval=5):
    api_key = get_api_key(model)
    client = OpenAI(api_key=api_key)
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(model=model, messages=messages)
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except Exception as e:
            print(f"[Retry] GPT error: {e}")
            time.sleep(retry_interval)
    return False

def call_ds(model, messages, retry_num=5, retry_interval=5):
    api_key = get_api_key(model)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(model=model, messages=messages)
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except Exception as e:
            print(f"[Retry] GPT error: {e}")
            time.sleep(retry_interval)
    return False

def call_qwen2(model, messages, retry_num=10, retry_interval=5):
    dashscope.api_key = get_api_key(model)
    for _ in range(retry_num):
        try:
            response = dashscope.Generation.call(model=model, messages=messages, enable_search=False)
            text = response.output.text
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
    return False

import time
from openai import OpenAI

def call_model_openai_api(
    model,
    messages,
    retry_num=30,
    retry_interval=10,
    thinking=False,
    response_format=None,
    search=False,
    quota_retry_num=20,
    temperature=1.0,
    n=1,
    max_n = 1, 
    stream=False
):
    from openai import OpenAI
    import time
      # OpenAI官方目前支持最大n=4
    if n > 1:
        stream = False
    client = OpenAI(
        api_key=get_api_key(model),
        base_url=get_url(model)
    )
    err_str = ""
    extra_body = {}
    if any(ali in model for ali in aliapi_model):
        if thinking:
            extra_body["enable_thinking"] = True
            extra_body["incremental_output"] = True
            stream = True
        else:
            extra_body["enable_thinking"] = False
        if search:
            extra_body["enable_search"] = True
        if response_format:
            extra_body["response_format"] = response_format
    if any(k in model for k in local_sft_model):
        # chat 模板开关（如果你的 tokenizer/chat_template 支持该 kwarg）
        extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}

    results = []
    total_need = n
    while total_need > 0:
        curr_n = min(max_n, total_need)
        normal_retries = 0
        quota_retries = 0
        while normal_retries < retry_num and quota_retries < quota_retry_num:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=extra_body,
                    temperature=temperature,
                    n=curr_n,
                    stream=stream
                )

                if stream:
                    # 只支持n=1的流式
                    reasoning_content = ""
                    answer_content = ""
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        reasoning_content += getattr(delta, "reasoning_content", "") or ""
                        answer_content += getattr(delta, "content", "") or ""
                    return reasoning_content.strip(), answer_content.strip()

                else:
                    # 批量采样
                    for choice in completion.choices:
                        content = getattr(choice.message, "content", "")
                        reasoning_content = getattr(choice.message, "reasoning_content", "")
                        if any(k in model for k in local_sft_model):
                            reasoning_content = extract_reasoning_block(content)
                            content = extract_answer_block(content)

                        results.append((reasoning_content.strip(), content.strip()))
                            # break
                    break  # 当前批次采样完成，跳出本批次重试循环

            except Exception as e:
                err_str = str(e).lower()
                print(err_str)
                if "input length should" in err_str:
                    length = 0
                    for message in messages:
                        prompt = message.get("content")
                        length += len(enc.encode(prompt, disallowed_special=()))
                        length += 2
                    print(f"[Input Length Error] Input length is {length} tokens, which exceeds the model's limit.", "api error:", e)
                    normal_retries = retry_num + 1
                    return f"error:{err_str}", f"error:{err_str}"
                if "quota" in err_str:
                    quota_retries += 1
                    print(f"[Quota Retry {quota_retries}] API error: {e}. Retrying in {retry_interval}s...")
                elif "inappropriate content" in err_str:
                    normal_retries = retry_num + 1
                    print(f"[Inappropriate Content] API error: {e}. Stopping further retries.")
                    # break
                elif "timed out" in err_str.lower():
                    normal_retries += retry_num + 1
                    print(f"[RequestTimeout] API error: {e}. Stop retry.")
                    # break
                else:
                    normal_retries += 1
                    print(f"[Normal Retry {normal_retries}] API error: {e}. Retrying in {retry_interval}s...")
                
                if normal_retries < retry_num:
                    time.sleep(retry_interval)
        else:
            print("[ERROR] Reached maximum retry attempts in batch.")
            return f"error:{err_str}", f"error:{err_str}"

        total_need -= curr_n

    reasons = [item[0] for item in results[:n]]
    answers = [item[1] for item in results[:n]]
    
    if n == 1:
        return reasons[0], answers[0]
    else:
        return reasons, answers  # 返回前n条（防止多采）

def call_qwen3(model, messages, retry_num=10, retry_interval=10, thinking=False, response_format=None, search=False):
    dashscope.api_key = get_api_key(model)
    for i in range(retry_num):
        try:
            
            if thinking:
                ## todo
                completion = dashscope.Generation.call(
                    model=model,
                    messages=messages,
                    result_format="message" ,
                    enable_thinking=True,
                    stream=True,
                    incremental_output=True,
                    enable_search=False, 
                    response_format=response_format
                )

                reasoning_content = ""
                answer_content = ""
                is_answering = False

                for chunk in completion:
                    answer_content += chunk.output.choices[0].message.content

                # for chunk in completion:
                #     choice = chunk.output.choices[0].message
                #     reasoning = choice.reasoning_content
                #     answer = choice.content

                #     if not reasoning and not answer:
                #         continue

                #     if reasoning and not answer:
                #         # print(reasoning, end="", flush=True)
                #         reasoning_content += reasoning

                #     elif answer:
                #         if not is_answering:
                #             # print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                #             is_answering = True
                #         # print(answer, end="", flush=True)
                #         answer_content += answer

                # print("\n" + "=" * 20 + "思考结束" + "=" * 20)
                return answer_content.strip()
            else:
                
                responses = dashscope.Generation.call(model=model, messages=messages, enable_search=False, enable_thinking=False, result_format="message", response_format=response_format)
                
                text = responses.output.choices[0].message.content
                    # print(response.output.choices[0].message.content)
                return text.strip()
                
        except Exception as e:
            print(f"[Retry {i}] Qwen3 error: {e}")
            if i == retry_num - 1:
                print("[ERROR] Reached maximum retry attempts.")
                raise e
            time.sleep(retry_interval)
            
    return False

def call_model_dashscope_api(
    model,
    messages,
    retry_num=10,
    retry_interval=10,
    thinking=False,
    response_format=None,
    search=False,
    quota_retry_num=3,
    temperature=1.0,
    n=1,
    max_n=4,
    stream=False
):
    import dashscope
    import time
    from http import HTTPStatus
    
    # DashScope 目前支持最大n=4
    if n > 1:
        stream = False
    
    dashscope.api_key = get_api_key(model)
    err_str = ""
    
    results = []
    total_need = n
    
    while total_need > 0:
        curr_n = min(max_n, total_need)
        normal_retries = 0
        quota_retries = 0
        
        while normal_retries < retry_num and quota_retries < quota_retry_num:
            try:
                if stream or thinking:
                    # 流式调用
                    completion = dashscope.Generation.call(
                        model=model,
                        messages=messages,
                        result_format="message",
                        enable_thinking=thinking,
                        stream=True,
                        incremental_output=thinking,
                        enable_search=search,
                        temperature=temperature,
                        n=curr_n
                    )

                    if stream:
                        # 只支持n=1的流式
                        reasoning_content = ""
                        answer_content = ""
                        for chunk in completion:
                            if chunk.status_code != HTTPStatus.OK:
                                raise Exception(f"Stream error: {chunk}")
                            
                            choice = chunk.output.choices[0]
                            reasoning_content += getattr(choice.message, "reasoning_content", "") or ""
                            answer_content += getattr(choice.message, "content", "") or ""
                        return reasoning_content.strip(), answer_content.strip()
                else:
                    # 批量采样
                    responses = dashscope.Generation.call(
                        model=model,
                        messages=messages,
                        result_format="message",
                        enable_search=search,
                        enable_thinking=False,
                        temperature=temperature,
                        n=curr_n
                    )

                    print("___________")
                    print("call params:")
                    print(f"model: {model}")
                    print(f"messages: {messages}")
                    print(f"retry_num: {retry_num}")
                    print(f"retry_interval: {retry_interval}")
                    print(f"thinking: {thinking}")
                    print(f"response_format: {response_format}")
                    print(f"search: {search}")
                    print(f"quota_retry_num: {quota_retry_num}")
                    print(f"temperature: {temperature}")
                    print(f"n: {n}")
                    print(f"max_n: {max_n}")
                    print(f"stream: {stream}")
                    if responses.status_code != HTTPStatus.OK:
                        raise Exception(f"API error: {responses}")
                    
                    for choice in responses.output.choices:
                        content = getattr(choice.message, "content", "")
                        reasoning_content = getattr(choice.message, "reasoning_content", "")
                        results.append((reasoning_content.strip(), content.strip()))
                    break  # 当前批次采样完成，跳出本批次重试循环
            
            except Exception as e:
                err_str = str(e).lower()
                if "input length should" in err_str:
                    length = 0
                    for message in messages:
                        prompt = message.get("content")
                        length += len(enc.encode(prompt, disallowed_special=()))
                        length += 2
                    print(f"[Input Length Error] Input length is {length} tokens, which exceeds the model's limit.", "api error:", e)
                    normal_retries = retry_num + 1
                    return f"error:{err_str}", f"error:{err_str}"
                if "quota" in err_str:
                    quota_retries += 1
                    print(f"[Quota Retry {quota_retries}] API error: {e}. Retrying in {retry_interval}s...")
                elif "inappropriate content" in err_str:
                    normal_retries = retry_num + 1
                    print(f"[Inappropriate Content] API error: {e}. Stopping further retries.")
                else:
                    normal_retries += 1
                    print(f"[Normal Retry {normal_retries}] API error: {e}. Retrying in {retry_interval}s...")
                time.sleep(retry_interval)
        else:
            print("[ERROR] Reached maximum retry attempts in batch.")
            return f"error:{err_str}", f"error:{err_str}"
        
        total_need -= curr_n
    
    reasons = [item[0] for item in results[:n]]
    answers = [item[1] for item in results[:n]]
    
    if n == 1:
        return reasons[0], answers[0]
    else:
        return reasons, answers  # 返回前n条（防止多采）

def use_thinking_for_model(model):
    if "thinking" in model or "r1" in model or "glm-4.5" in model or "gemini-2.5" in model:
        return [True]

    elif "instruct" in model:
        return [False]
    elif "qwen3" in model or "claude-4" in model:
        return [False, True]
    else:
        return [False]

def call_model(messages, eval_model, thinking=False, response_format=None, search=False, dummy_call=False,best_of_n=1, temperature=0):
    if dummy_call:
        return "dummy call reason", "dummy call answer"

    return call_model_openai_api(eval_model, messages, thinking=thinking, response_format=response_format, search=search, n=best_of_n, temperature=temperature)
def get_decision_prompt(prompt_builder, prompt_version, full_question, title, doc_type, doc_len_tokens, context_snippet, answer_model, task_type, rag_chunks, head_n, rag_config):
    """
    根据提示版本生成决策提示

    :param prompt_builder: 提示构建函数
    :param prompt_version: 提示版本
    :param full_question: 完整问题
    :param title: 文档标题
    :param doc_type: 文档类型
    :param doc_len_tokens: 文档token长度
    :param context_snippet: 文档上下文片段
    :param answer_model: 回答模型
    :param task_type: 任务类型
    :param rag_chunks: RAG块
    :param head_n: 头部截取长度
    :return: 决策提示
    """
    base_params = {
        "query": full_question,
        "doc_title": title,
        "doc_type": doc_type,
        "doc_len_tokens": doc_len_tokens,
        "doc_head": context_snippet,
        "model": answer_model,
        "task_type": task_type + title,
    }
    
    base_params["head_n"] = head_n
    base_params["rag_config"] = rag_config
    return prompt_builder(**base_params)
def select_prompt_builder(prompt_version):
    mapping = {
        "preroute": build_decision_prompt_preroute,
    }
    return mapping.get(prompt_version, build_decision_prompt_preroute)

# MODEL_CONTEXT_WINDOW = {
#     "claude-3-5-haiku-20241022": 200000,
#     "claude-3-5-sonnet-20241022": 200000,
#     "claude-3-5-haiku": 200000,
#     "claude-3-5-sonnet": 200000,
#     "gemini-1.5-pro": 1000000,
#     "gpt-4o": 128000,
#     "gpt-4o-0806": 128000,
#     "gpt-4o-mini": 128000,
#     "llama-3.3-70b": 128000,
#     "llama-3.3-70b-turbo": 128000,
#     "llama-3.1-8b": 128000,
#     "llama-3.2-3b": 128000,
#     "mistral-nemo": 128000,
#     "qwen2.5-7b-instruct": 32768,
#     "qwen2.5-72b-instruct": 131072,
#     "qwen2.5-1m": 1000000,
# }

# def get_model_context_window(model_name: str) -> int:
#     return MODEL_CONTEXT_WINDOW.get(model_name, 128000)
def compare_two_answers(pred_a, pred_b, label, query, query_type, model="qwen-max", thinking=False):
    if query_type == 'hallu':
        prompt = f"""You will be given a question, a groundtruth answer, and two answers from AI assistants (Assistant A and Assistant B). 
This question may not be directly answerable from the source text.

Your task:
1. Check whether each assistant is hallucinating. If the answer is **consistent** with the groundtruth, and the question is **not present in the original text**, it is considered **not hallucinating**.
2. For each answer, assign a quality score from 1 to 4:
   - 4 = Fully correct (not hallucinating and consistent)
   - 3 = Mostly correct (minor detail missing or uncertain)
   - 2 = Partially correct (some correct points, some hallucination)
   - 1 = Hallucinated, irrelevant, or incorrect
3. Write a brief comparative analysis.
4. Decide who is better: "A", "B", or "Tie"

Return JSON in this format:
{{
  "analysis": "...",
  "score_a": int,
  "score_b": int,
  "better": "A" | "B" | "Tie"
}}

[Question]
{query}

[Groundtruth Answer]
{label}

[Assistant A's Answer]
{pred_a}

[Assistant B's Answer]
{pred_b}

Start your evaluation:"""

    else:
        prompt = f"""You will be given a question, a groundtruth answer, and two AI assistant answers.

Your task:
1. Judge each answer's factual accuracy and completeness relative to the groundtruth.
2. For each answer, assign a score from 1 to 4:
   - 4 = Fully correct (covers all key points, no factual or logical flaws)
   - 3 = Mostly correct (core is right, minor flaws or omissions)
   - 2 = Partially correct (some accurate parts, but notable errors or gaps)
   - 1 = Incorrect or hallucinated (serious flaws, irrelevant, fabricated)
3. Provide a brief comparison explaining your scoring.
4. Choose which assistant is better: "A", "B", or "Tie"

Return your result in the following JSON format:
{{
  "analysis": "...",
  "score_a": int,
  "score_b": int,
  "better": "A" | "B" | "Tie"
}}

[Question]
{query}

[Groundtruth Answer]
{label}

[Assistant A's Answer]
{pred_a}

[Assistant B's Answer]
{pred_b}

Start your evaluation:"""

    messages = [
        {
            "role": "system",
            "content": "You are an impartial evaluator. Score two assistant answers independently on a 1–4 scale and explain your judgment."
        },
        {"role": "user", "content": prompt},
    ]

    for i in range(3):
        try:
            reason, response = call_model(eval_model=model, messages=messages, response_format={'type': 'json_object'}, thinking=thinking)
            # print(f"[DEBUG] Comparison response: {response}")

            import json
            result = json.loads(response)
            if all(k in result for k in ("analysis", "score_a", "score_b", "better")):
                return {
                    "analysis": result["analysis"],
                    "score_a": int(result["score_a"]),
                    "score_b": int(result["score_b"]),
                    "better": result["better"]
                }
        except Exception as e:
            print(f"[Compare Retry {i}] {e}")
            if i == 2:
                print("[ERROR] Failed to get a valid comparison response after multiple retries.")
                print("[ERROR] Response content:", response)
                raise e
            time.sleep(3)

    return {
        "analysis": "Model failed to evaluate.",
        "score_a": 1,
        "score_b": 1,
        "better": "Tie"
    }

import re
MODEL_MAX_TOKENS_PATTERNS = [
    # Qwen 系列
    (r"^qwen-max",     {"thinking": 98000,      "non_thinking": 128000}),
    (r"^qwen-plus",    {"thinking": 98000,      "non_thinking": 128000}),
    (r"^deepseek-r1",    {"thinking": 98000,      "non_thinking": 98000}),
    (r"^qwen3-1.7b",   {"thinking": 28000,      "non_thinking": 30000}),
    (r"^qwen3-0.6b",   {"thinking": 28000,      "non_thinking": 30000}),
    (r"^qwen3",        {"thinking": 98000,      "non_thinking": 128000}),
    (r"^qwen-long",    {"thinking": 10_000_000, "non_thinking": 10_000_000}),
    # Claude 3.5/3.x 系列
    (r"^claude-3\.5",  {"thinking": 200_000,    "non_thinking": 200_000}),
    (r"^claude-3",     {"thinking": 200_000,    "non_thinking": 200_000}),
    # Gemini 系列
    (r"^gemini-1\.5-pro", {"thinking": 1_000_000, "non_thinking": 1_000_000}),
    (r"^gemini-2\.5", {"thinking": 1_000_000, "non_thinking": 1_000_000}),
    (r"^gemini-advanced", {"thinking": 1_000_000, "non_thinking": 1_000_000}),
    # GPT-4/4o/4.1 系列
    (r"^gpt-4o",       {"thinking": 128_000,    "non_thinking": 128_000}),
    (r"^gpt-4\.1",     {"thinking": 1_000_000,  "non_thinking": 1_000_000}),
    # Llama 4 系列（全量上下文）
    (r"^llama-4",      {"thinking": 10_000_000, "non_thinking": 10_000_000}),
    # Llama 3/3.1/3.2/3.3 可补充，如下为默认128K（如有更新可补充）
    (r"^llama-3",      {"thinking": 128_000,    "non_thinking": 128_000}),
    # Mistral Nemo、Mixtral等
    (r"^mistral-nemo", {"thinking": 128_000,    "non_thinking": 128_000}),
    # 兜底fallback，128K
]
def get_max_token_length(model_name: str, thinking: bool = False) -> int:
    base_name = model_name.lower().split("/")[-1]  # 去掉路径前缀或 checkpoint 子目录
    for pattern, config in MODEL_MAX_TOKENS_PATTERNS:
        if re.match(pattern, base_name):
            return config["thinking" if thinking else "non_thinking"]
    return 128000  # 默认 fallback

# def describe_doc_length(doc_len_tokens: int) -> str:
#     if doc_len_tokens < 4000:
#         return f"{doc_len_tokens} tokens (very short)"
#     elif doc_len_tokens < 16000:
#         return f"{doc_len_tokens} tokens (short)"
#     elif doc_len_tokens < 40000:
#         return f"{doc_len_tokens} tokens (medium length)"
#     elif doc_len_tokens < 80000:
#         return f"{doc_len_tokens} tokens (long)"
#     else:
#         return f"{doc_len_tokens} tokens (very long)"

def plot_score_distributions(json_data, save_path, router_model, prompt_version, head_n, best_of_n):

    json_data = expand_example_field(json_data)
    """绘制分数分布图"""
    # 准备数据
    equal_scores_rag = []
    equal_scores_lc = []
    rag_better_scores_rag = []
    rag_better_scores_lc = []
    lc_better_scores_rag = []
    lc_better_scores_lc = []
    
    for ex in json_data:
        rag_score = ex["score_rag"]
        lc_score = ex["score_full"]
        
        if rag_score == lc_score:
            equal_scores_rag.append(rag_score)
            equal_scores_lc.append(lc_score)
        elif rag_score > lc_score:
            rag_better_scores_rag.append(rag_score)
            rag_better_scores_lc.append(lc_score)
        else:  # lc_score > rag_score
            lc_better_scores_rag.append(rag_score)
            lc_better_scores_lc.append(lc_score)
    print("RAG > LC:", len(rag_better_scores_rag))
    print("LC > RAG:", len(lc_better_scores_rag))
    print("RAG = LC:", len(equal_scores_rag))

    # 创建保存图表的目录
    plot_dir = os.path.dirname(save_path)
    plot_save_path = os.path.join(plot_dir, f"score_distributions_{router_model}_{prompt_version}_{head_n}_bon{best_of_n}.png")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 相等部分的分数分布图
    if equal_scores_rag:
        axes[0].scatter(equal_scores_rag, equal_scores_lc, alpha=0.6, color='blue')
        axes[0].set_xlabel('RAG Score')
        axes[0].set_ylabel('Long Context Score')
        axes[0].set_title('Score Distribution (Equal)')
        axes[0].grid(True)
    else:
        axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Score Distribution (Equal)')
    
    # rag>lc的分数分布图
    if rag_better_scores_rag:
        axes[1].scatter(rag_better_scores_rag, rag_better_scores_lc, alpha=0.6, color='green')
        axes[1].set_xlabel('RAG Score')
        axes[1].set_ylabel('Long Context Score')
        axes[1].set_title('Score Distribution (RAG > LC)')
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Score Distribution (RAG > LC)')
    
    # lc>rag的分数分布图
    if lc_better_scores_rag:
        axes[2].scatter(lc_better_scores_rag, lc_better_scores_lc, alpha=0.6, color='red')
        axes[2].set_xlabel('RAG Score')
        axes[2].set_ylabel('Long Context Score')
        axes[2].set_title('Score Distribution (LC > RAG)')
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Score Distribution (LC > RAG)')
    
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 分数分布图已保存到: {plot_save_path}")

def compute_metrics_by_strategy(json_data, strategy_func):
    """根据指定的策略函数计算指标"""
    from collections import defaultdict

    metrics = {
        "accuracy": 0.0,
        "auc": 0.0,
        "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "total_examples": 0,
        "type_accuracy": defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0}),
        "level_accuracy": defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0}),
        "context_length_accuracy": defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0}),
        "classification_report": "",
        "actual_score_avg": 0.0  # 添加实际得分平均值
    }
    
    y_true, y_pred = [], []
    valid_examples = []
    actual_scores = []  # 用于计算实际路由策略得分

    for example in json_data:
        ex = example["example"] if "example" in example else example
        
        if "score_rag" not in ex:
            ex["score_rag"] = 1 if ex.get("rag_correct") else 0
            ex["score_full"] = 1 if ex.get("lc_correct") else 0

        # 获取更好的策略
        if strategy_func.__name__ == "define_better_strategy_equal_included":
            # tied scores: accept the router's own choice
            rag_score = ex["score_rag"]
            lc_score = ex["score_full"]
            
            if rag_score > lc_score:
                better_strategy = "rag"
            elif rag_score < lc_score:
                better_strategy = "full"
            else:
                # 相等时，根据路由决策确定
                route_decision = ex.get("route_decision", "")
                # 无论选rag还是full都算对
                better_strategy = route_decision  # 使用实际的路由决策作为正确标准
        else:
            # 原有逻辑
            better_strategy = strategy_func(ex)
            if better_strategy == "equal":
                continue
                
        # 处理BON场景：routes是列表
        route_decisions = ex.get("route_decision", [])
        if isinstance(route_decisions, list):
            # BON场景：检查是否有better_decision，否则使用多数投票
            if route_decisions:
                # 统一处理long_context为full
                normalized_routes = ["full" if r in ["long_context", "full"] else r for r in route_decisions]
                
                # 检查是否有better_decision
                if better_strategy in normalized_routes:
                    route_decision = better_strategy
                else:
                    opp_strategy = "rag" if better_strategy == "full" else "full"
                    route_decision = opp_strategy
            else:
                continue  # 没有有效路由决策
        else:
            # 单路由决策场景
            route_decision = str(route_decisions)
            if route_decision in ["long_context", "full"]:
                route_decision = "full"
        
        # 确保better_strategy有效
        if better_strategy not in ["rag", "full"]:
            continue
            
        y_true.append(1 if better_strategy == "rag" else 0)
        y_pred.append(1 if route_decision == "rag" else 0)
        valid_examples.append(ex)
        
        # 计算实际路由策略得分
        if route_decision == "rag":
            actual_scores.append(ex["score_rag"])
        else:  # full or long_context
            actual_scores.append(ex["score_full"])
        
        # 计算是否选对
        correct = route_decision == better_strategy
        
        # 更新各种统计
        if "context_length" in ex:
            type_, level, context_length = ex.get("type", ""), ex.get("level", ""), ex.get("context_length", "")
            metrics["type_accuracy"][type_]["correct"] += int(correct)
            metrics["type_accuracy"][type_]["total"] += 1
            metrics["level_accuracy"][level]["correct"] += int(correct)
            metrics["level_accuracy"][level]["total"] += 1
            metrics["context_length_accuracy"][context_length]["correct"] += int(correct)
            metrics["context_length_accuracy"][context_length]["total"] += 1
        
        # 统计选择情况
        if route_decision == "rag":
            metrics["rag_count"] = metrics.get("rag_count", 0) + 1
            # 统计选择 RAG 时，真实标签的情况
            if better_strategy == "rag":
                metrics["rag_when_rag"] = metrics.get("rag_when_rag", 0) + 1
            else:
                metrics["rag_when_lc"] = metrics.get("rag_when_lc", 0) + 1
        elif route_decision == "full":
            metrics["lc_count"] = metrics.get("lc_count", 0) + 1
            # 统计选择 LC 时，真实标签的情况
            if better_strategy == "full":
                metrics["lc_when_lc"] = metrics.get("lc_when_lc", 0) + 1
            else:
                metrics["lc_when_rag"] = metrics.get("lc_when_rag", 0) + 1
        
        # 统计选对和选错的次数
        if correct:
            metrics["correct_count"] = metrics.get("correct_count", 0) + 1
        else:
            metrics["incorrect_count"] = metrics.get("incorrect_count", 0) + 1

    if valid_examples:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        try:
            metrics["auc"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            metrics["auc"] = 0.0
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics["confusion_matrix"] = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
        metrics["total_examples"] = len(valid_examples)
        metrics["classification_report"] = classification_report(y_true, y_pred, target_names=["full", "rag"])
        
        # 计算实际得分平均值
        if actual_scores:
            metrics["actual_score_avg"] = sum(actual_scores) / len(actual_scores)
        else:
            metrics["actual_score_avg"] = 0.0
        
        # 计算准确率
        for d in [metrics["type_accuracy"], metrics["level_accuracy"], metrics["context_length_accuracy"]]:
            for key in d:
                d[key]["accuracy"] = d[key]["correct"] / d[key]["total"] if d[key]["total"] > 0 else 0.0
    
    return metrics

def define_better_strategy(example):
    if example.get("score_rag", 1 if example.get("rag_correct") else 0) >= example.get("score_full", 1 if example.get("lc_correct") else 0):
        return "rag"
    else:
        return "full"

def define_better_strategy_exequal(example):
    """Ideal path label; returns 'equal' on tied scores (tied samples excluded)."""
    if example["score_rag"] > example["score_full"]:
        return "rag"
    elif example["score_rag"] < example["score_full"]:
        return "full"
    else:
        # 相等时，根据路由决策确定
        return "equal"

def define_better_strategy_strict(example):
    """Ideal path label; ties default to RAG."""
    if example["score_rag"] >= example["score_full"]:
        return "rag"
    else:
        return "full"

def define_better_strategy_equal_included(example):
    """Ideal path label; on tied scores either choice counts as correct."""
    rag_score = example["score_rag"]
    lc_score = example["score_full"]
    
    if rag_score > lc_score:
        return "rag"
    elif rag_score < lc_score:
        return "full"
    else:
        # 相等时，根据路由决策确定
        return "rag"  # 返回一个默认值，实际在compute_metrics_by_strategy中会根据route_decision判断
def expand_example_field(data):
    new_data = []
    for item in data:
        item_copy = item.copy()
        example = item_copy.pop("example", {})
        if isinstance(example, dict):
            item_copy.update(example)
        new_data.append(item_copy)
    return new_data

def compute_and_print_metrics(json_data):
    # compute metrics under the three tie-handling settings
    json_data = expand_example_field(json_data)
    
    # 添加过滤逻辑，移除包含 error 字样的项
    filtered_data = []
    for item in json_data:
        # 检查 full 和 rag 字段中是否包含 "error" 字样
        has_error = False
        if isinstance(item.get('full'), str) and 'error' in item['full'].lower():
            has_error = True
        if isinstance(item.get('rag'), str) and 'error' in item['rag'].lower():
            has_error = True
        
        # 如果没有 error，才添加到过滤后的数据中
        if not has_error:
            filtered_data.append(item)
    json_data = filtered_data
    print(f"过滤后剩余样本数: {len(json_data)}")
    print("\n=== 排除相等样本 ===")
    strict_metrics = compute_metrics_by_strategy(json_data, define_better_strategy_exequal)
    print(f"- Accuracy: {strict_metrics['accuracy']:.4f}")
    print(f"- AUC: {strict_metrics['auc']:.4f}")
    print(f"- Total Examples: {strict_metrics['total_examples']}")
    print("- Confusion Matrix:")
    print(strict_metrics["confusion_matrix"])
    print("- Classification Report:")
    print(strict_metrics["classification_report"])
    print("- 实际路由策略得分:")
    print("Actual Route Score:", strict_metrics["actual_score_avg"])

    print("\n=== 包含相等样本(相等rag优先) ===")
    ragtie_metrics = compute_metrics_by_strategy(json_data, define_better_strategy_strict)
    print(f"- Accuracy: {ragtie_metrics['accuracy']:.4f}")
    print(f"- AUC: {ragtie_metrics['auc']:.4f}")
    print(f"- Total Examples: {ragtie_metrics['total_examples']}")
    print("- Confusion Matrix:")
    print(ragtie_metrics["confusion_matrix"])
    print("- Classification Report:")
    print(ragtie_metrics["classification_report"])
    print("- 实际路由策略得分:")
    print("Actual Route Score:", ragtie_metrics["actual_score_avg"])

    print("\n=== Include ties (either choice correct) ===")
    equal_included_metrics = compute_metrics_by_strategy(json_data, define_better_strategy_equal_included)
    print(f"- Accuracy: {equal_included_metrics['accuracy']:.4f}")
    print(f"- AUC: {equal_included_metrics['auc']:.4f}")
    print(f"- Total Examples: {equal_included_metrics['total_examples']}")
    print("- Confusion Matrix:")
    print(equal_included_metrics["confusion_matrix"])
    print("- Classification Report:")
    print(equal_included_metrics["classification_report"])
    print("- 实际路由策略得分:")
    print("Actual Route Score:", equal_included_metrics["actual_score_avg"])
    # 计算相等样本数量
    equal_count = 0
    for example in json_data:
        ex = example["example"] if "example" in example else example
        if "score_rag" not in ex:
            ex["score_rag"] = 1 if ex.get("rag_correct") else 0
            ex["score_full"] = 1 if ex.get("lc_correct") else 0
        
        if ex["score_rag"] == ex["score_full"]:
            equal_count += 1
    
    print(f"\n--- 统计汇总 ---")
    print(f"- 总样本数: {len(json_data)}")
    print(f"- 相等样本数: {equal_count}")
    print(f"- exclude-tied valid samples: {strict_metrics['total_examples']}")
    print(f"- ties->RAG valid samples: {ragtie_metrics['total_examples']}")
    print(f"- either-choice valid samples: {equal_included_metrics['total_examples']}")

    full_scores = [ex["score_full"] for ex in json_data]
    rag_scores = [ex["score_rag"] for ex in json_data]
    better_scores = [ex["score_rag"] if define_better_strategy(ex) == "rag"
                     else ex["score_full"] for ex in json_data]

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    print("\n--- Strategy Comparison (Global) ---")
    print(f"If All FULL:           {avg(full_scores):.4f}")
    print(f"If All RAG:            {avg(rag_scores):.4f}")
    print(f"If All BETTER:         {avg(better_scores):.4f}")
    
    # 计算实际路由策略的分数
    actual_scores = []
    for ex in json_data:
        route = ex.get("route_decision", "full")
        if route == "rag":
            actual_scores.append(ex["score_rag"])
        else:  # full or long_context
            actual_scores.append(ex["score_full"])
    print(f"Actual Route Strategy: {avg(actual_scores):.4f}")

    avg_rag_scores = avg(rag_scores)
    avg_full_scores = avg(full_scores)
    avg_better_scores = avg(better_scores)
    avg_actual_scores = avg(actual_scores)

    # 输出详细的分组统计
    print("\n--- 按文档类型统计 ---")
    for k, v in strict_metrics["type_accuracy"].items():
        print(f"- {k} (exclude-tied): {v['accuracy']:.4f} ({v['correct']}/{v['total']})")
    for k, v in ragtie_metrics["type_accuracy"].items():
        print(f"- {k} (ties->RAG): {v['accuracy']:.4f} ({v['correct']}/{v['total']})")
    for k, v in equal_included_metrics["type_accuracy"].items():
        print(f"- {k} (either): {v['accuracy']:.4f} ({v['correct']}/{v['total']})")

    print("\n--- 按问题级别统计 ---")
    for k, v in strict_metrics["level_accuracy"].items():
        print(f"- {k} (exclude-tied): {v['accuracy']:.4f} ({v['correct']}/{v['total']})")
    for k, v in ragtie_metrics["level_accuracy"].items():
        print(f"- {k} (ties->RAG): {v['accuracy']:.4f} ({v['correct']}/{v['total']})")
    for k, v in equal_included_metrics["level_accuracy"].items():
        print(f"- {k} (either): {v['accuracy']:.4f} ({v['correct']}/{v['total']})")
    
    equal_included_metrics["avg_rag_scores"] = avg_rag_scores
    equal_included_metrics["avg_full_scores"] = avg_full_scores
    equal_included_metrics["avg_better_scores"] = avg_better_scores
    equal_included_metrics["avg_actual_scores"] = avg_actual_scores

    # 打印选择情况统计
    print(f"\n--- 选择情况统计 ---")
    print(f"- 选择 RAG 的次数: {equal_included_metrics.get('rag_count', 0)}")
    print(f"  - 其中真实值是 RAG 的次数: {equal_included_metrics.get('rag_when_rag', 0)}")
    print(f"  - 其中真实值是 LC 的次数: {equal_included_metrics.get('rag_when_lc', 0)}")
    print(f"- 选择 LC 的次数: {equal_included_metrics.get('lc_count', 0)}")
    print(f"  - 其中真实值是 LC 的次数: {equal_included_metrics.get('lc_when_lc', 0)}")
    print(f"  - 其中真实值是 RAG 的次数: {equal_included_metrics.get('lc_when_rag', 0)}")
    
    # 打印各个标准下选对和选错的次数
    print(f"\n--- Correct/incorrect (exclude-tied) ---")
    print(f"- 选对次数: {strict_metrics.get('correct_count', 0)}")
    print(f"- 选错次数: {strict_metrics.get('incorrect_count', 0)}")
    
    print(f"\n--- Correct/incorrect (ties->RAG) ---")
    print(f"- 选对次数: {ragtie_metrics.get('correct_count', 0)}")
    print(f"- 选错次数: {ragtie_metrics.get('incorrect_count', 0)}")
    
    print(f"\n--- Correct/incorrect (either) ---")
    print(f"- 选对次数: {equal_included_metrics.get('correct_count', 0)}")
    print(f"- 选错次数: {equal_included_metrics.get('incorrect_count', 0)}")
    
    return strict_metrics, ragtie_metrics, equal_included_metrics

def extract_reasoning_block(text: str) -> str:
    start_match = re.search(r"<think>", text, re.IGNORECASE)
    if not start_match:
        return ""
        
    start_index = start_match.end()
    text_after_start = text[start_index:]
    
    end_match = re.search(r"</think>", text_after_start, re.IGNORECASE)
    if end_match:
        return text_after_start[:end_match.start()].strip()
    else:
        return text_after_start.strip()

def extract_answer_block(text: str) -> str:
    """提取 </think> 之后的内容（即最终回答部分）"""
    end_match = re.search(r"</think>", text, re.IGNORECASE)
    if not end_match:
        # 如果没有 </think>，那就把整个 text 当成答案
        return text.strip()
    # 提取结束标签之后的内容
    return text[end_match.end():].strip()

def extract_thinking_block(text: str) -> str:
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()
import re

import re
def extract_decision_from_thinking_no_both(thinking: str) -> str:
    route = None

    # --- 1) 原规则（保持不变） ---
    match = re.search(r"<decision>\s*(rag|long_context|full)", thinking, re.IGNORECASE)
    if match and route is None:
        route = match.group(1).strip().lower()
    
    # --- 2) 放宽规则 ---
    if route is None:
        tolerant_pattern = (
            r"(?:\*{0,3}\s*`{0,3}\s*)<\s*decision\s*>"
            r"(?:\s*`{0,3}\s*\*{0,3})?"
            r"\s*[:\-–]?\s*"
            r"(?:\r?\n\s*)?"
            r"(rag|long[_\-\s]*context|full)\b"
        )
        m = re.search(tolerant_pattern, thinking, re.IGNORECASE | re.DOTALL)
        if m:
            v = m.group(1).strip().lower().replace(" ", "").replace("-", "_")
            route = v
    
    # --- 3) 兜底：轻度清洗 ---
    if route is None:
        cleaned = re.sub(r"[*`]+", "", thinking)
        cleaned = re.sub(r"\s+\n", "\n", cleaned) 
        cleaned = re.sub(r"\s+", " ", cleaned)    
        m = re.search(r"<\s*decision\s*>\s*[:\-–]?\s*(rag|long_context|full)\b", cleaned, re.IGNORECASE)
        if m:
            route = m.group(1).strip().lower()

    # --- 4) 大保底：找最后一个 <decision> 开标签 ---
    if route is None:
        open_tag_iter = list(re.finditer(r"<\s*decision\s*>", thinking, re.IGNORECASE))
        if open_tag_iter:
            start = open_tag_iter[-1].end()
            tail = thinking[start:start+800]
            m = re.search(r"(rag|long[_\-\s]*context|full)\b", tail, re.IGNORECASE)
            if m:
                route = m.group(1).lower()

    # --- 5) 再退一步：找最后一个 decision 单词 ---
    if route is None:
        idx = thinking.lower().rfind("decision")
        if idx != -1:
            if not re.match(r"</\s*decision\s*>", thinking[idx-2:idx+11].lower()):
                tail = thinking[idx:idx+800]
                m = re.search(r"(rag|long[_\-\s]*context|full)\b", tail, re.IGNORECASE)
                if m:
                    route = m.group(1).lower()

    # --- 6) 终极兜底：从后往前找 ---
    if route is None:
        m_all = list(re.finditer(r"(rag|long[_\-\s]*context|full)\b", thinking, re.IGNORECASE))
        if m_all:
            route = m_all[-1].group(1).lower()

    # === 统一后处理 ===
    if route is None:
        return "no_matching"

    # 归一化写法
    norm = route.replace(" ", "").replace("-", "_")
    if norm in {"longcontext", "long_context"}:
        route = "full"
    
    if route not in {"rag", "full"}:
        return "no_matching"

    return route

def extract_decision_from_thinking(thinking: str) -> str:
    route = None

    # --- 1) 原规则（保持不变） ---
    match = re.search(r"<decision>\s*(rag|long_context|full|both)", thinking, re.IGNORECASE)
    if match and route is None:
        route = match.group(1).strip().lower()
    
    # --- 2) 放宽规则 ---
    if route is None:
        tolerant_pattern = (
            r"(?:\*{0,3}\s*`{0,3}\s*)<\s*decision\s*>"
            r"(?:\s*`{0,3}\s*\*{0,3})?"
            r"\s*[:\-–]?\s*"
            r"(?:\r?\n\s*)?"
            r"(rag|long[_\-\s]*context|full|both)\b"
        )
        m = re.search(tolerant_pattern, thinking, re.IGNORECASE | re.DOTALL)
        if m:
            v = m.group(1).strip().lower().replace(" ", "").replace("-", "_")
            route = v
    
    # --- 3) 兜底：轻度清洗 ---
    if route is None:
        cleaned = re.sub(r"[*`]+", "", thinking)
        cleaned = re.sub(r"\s+\n", "\n", cleaned) 
        cleaned = re.sub(r"\s+", " ", cleaned)    
        m = re.search(r"<\s*decision\s*>\s*[:\-–]?\s*(rag|long_context|full|both)\b", cleaned, re.IGNORECASE)
        if m:
            route = m.group(1).strip().lower()

    # --- 4) 大保底：找最后一个 <decision> 开标签 ---
    if route is None:
        open_tag_iter = list(re.finditer(r"<\s*decision\s*>", thinking, re.IGNORECASE))
        if open_tag_iter:
            start = open_tag_iter[-1].end()
            tail = thinking[start:start+800]
            matches = re.findall(r"(rag|long[_\-\s]*context|full|both)\b", tail, re.IGNORECASE)
            if matches:
                route = matches[-1].lower()

    # --- 5) 再退一步：找最后一个 decision 单词 ---
    if route is None:
        idx = thinking.lower().rfind("decision")
        if idx != -1:
            if not re.match(r"</\s*decision\s*>", thinking[idx-2:idx+11].lower()):
                tail = thinking[idx:idx+800]
                matches = re.findall(r"(rag|long[_\-\s]*context|full|both)\b", tail, re.IGNORECASE)
                if matches:
                    route = matches[-1].lower()

    # --- 6) 终极兜底：从后往前找 ---
    if route is None:
        m_all = list(re.finditer(r"(rag|long[_\-\s]*context|full|both)\b", thinking, re.IGNORECASE))
        if m_all:
            route = m_all[-1].group(1).lower()

    # === 统一后处理 ===
    if route is None:
        return "no_matching"

    # 归一化写法
    norm = route.replace(" ", "").replace("-", "_")
    if norm in {"longcontext", "long_context"}:
        route = "full"
    elif norm == "both":
        print("⚠️ 路由决策为 both，统一处理为 rag")
        route = "rag"
    
    if route not in {"rag", "full"}:
        return "no_matching"

    return route

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_data(data_list: List[Dict], preds: Dict[str, Dict], key: str="_id") -> List[Dict]:
    merged = []

    if isinstance(preds, list):
        preds = {item[key]: item for item in preds}

    for item in data_list:
        _id = item[key]
        if _id in preds:
            merged_item = {**item, **preds[_id]}
            merged.append(merged_item)
        else:
            pass
            # print(f"⚠️ 缺少预测结果：{_id}")
    return merged

def save_json(path: str, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def describe_doc_length(doc_len_tokens: int) -> str:
    if doc_len_tokens < 4000:
        return f"{doc_len_tokens} tokens (very short)"
    elif doc_len_tokens < 16000:
        return f"{doc_len_tokens} tokens (short)"
    elif doc_len_tokens < 40000:
        return f"{doc_len_tokens} tokens (medium length)"
    elif doc_len_tokens < 80000:
        return f"{doc_len_tokens} tokens (long)"
    else:
        return f"{doc_len_tokens} tokens (very long)"

system_prompt = f"""
    You are a routing strategy evaluator for large language models. Your task is to determine, based on the given query and document metadata, whether the query should be answered using RAG (Retrieval-Augmented Generation) or LONG_CONTEXT (inputting the entire document as context).

Note: The strategy you choose will be executed by a downstream large language model. This model will perform the actual answering task, so your routing decision should consider how effectively this model can handle each strategy, given the document and query characteristics.
    """

RAG_CONFIG = {
    "chunk_size": 600,
    "chunk_overlap": 100,
    "embed_model": "embedding_models/gte-multilingual-base",
    "rerank_model": "embedding_models/gte-multilingual-reranker-base",
    "vector_ratio": 0.5,
    "rerank_size": 5,
}

def build_rag_config_str(rag_config: dict = None) -> str:
    """
    Build RAG configuration string from config dict.
    
    Args:
        rag_config: RAG configuration dict, will be merged with RAG_CONFIG
        
    Returns:
        Formatted RAG configuration string
    """
    # 合并rag_config和RAG_CONFIG，rag_config中的值优先级更高
    merged_config = RAG_CONFIG.copy()
    if rag_config:
        merged_config.update(rag_config)
    
    return f"""
## RAG Configuration
- chunk_size: {merged_config.get('chunk_size', 'unknown')} tokens
- chunk_overlap: {merged_config.get('chunk_overlap', 'unknown')} tokens
- embed_model: {merged_config.get('embed_model', 'unknown')}
- rerank_model: {merged_config.get('rerank_model', 'unknown')}
- vector_ratio: {merged_config.get('vector_ratio', 'unknown')}
- rerank_size: {merged_config.get('rerank_size', 'unknown')}
"""

def build_decision_prompt_preroute(
    query: str,
    doc_title: str,
    doc_type: str,
    doc_len_tokens: int,
    doc_head: str,
    model: str,
    task_type: str,
    thinking: bool = False,
    rag_config: dict = None,
    head_n: int = 200,
) -> str:
    """构造决策提示符 - 思维链版 + 预处理增强版（效率优先：结果相近时选RAG）"""

    length_str = describe_doc_length(doc_len_tokens)
    max_window_tokens = get_max_token_length(model, thinking=thinking)
    document_fits = "Yes" if doc_len_tokens <= max_window_tokens else "No"
    RAG_CONFIG_str = build_rag_config_str(rag_config)
    return f"""
## Inputs
- query: “{query}”
- task_type: {task_type}
- document_title: “{doc_title}”
- document_type: {doc_type}
- document_length: {length_str} ({doc_len_tokens} tokens)
- answering_model: {model}
- answering_max_window: {max_window_tokens} tokens
- document_fits_window: {document_fits}
- document_head content: “{doc_head[:head_n]}...”
{RAG_CONFIG_str}
---

## Instructions

You are tasked with choosing the most appropriate strategy — **RAG**, **LONG_CONTEXT** — for answering the user query, based on the characteristics of the query, document, and model.

Please complete the `<thinking>` block below. In each step, provide a clear judgment and explain how it affects your strategy choice.

1. `<step1>`: Identify the question type (e.g., factual, reasoning, comparison, judgment, etc.) and the document type (e.g., book, article, report). How do these affect the need for deep context understanding or precise retrieval?

2. `<step2>`: Assess whether the relevant information is likely concentrated in one part of the document or scattered across multiple sections. How does this affect strategy selection?

3. `<step3>`: Evaluate whether the document can fully fit into the context window, based on `document_fits_window` above. If not, how does that impact strategy choice?

4. `<step4>`: Consider whether the query can be answered through keyword-based retrieval (e.g., names, dates), or requires synthesizing implicit logic, analogies, or multi-part reasoning.

5. `<step5>`: Reflect on the model being used (e.g., {model}). Consider both its context window size and model capacity (parameters). Although some models may have large context windows, smaller models may still struggle with effective long-context reasoning due to limited capacity. How does this influence your strategy recommendation?

6. `<step6_efficiency>`: Compare **expected efficiency** of RAG vs LONG_CONTEXT: expected context size, retrieval selectivity, latency, and cost. If quality is likely similar, which strategy is more efficient?

7. `<reflection>`: Based on your reasoning above, state which strategy is more suitable overall — **RAG**, **LONG_CONTEXT** — and explain why. 

8. `<decision>`: Write your final strategy choice clearly as either `RAG`, `LONG_CONTEXT`. 

### Decision Rules (must be followed)
- If both strategies are **equally suitable** or quality difference is **negligible/uncertain**, **prefer RAG for efficiency**.
- Prefer **LONG_CONTEXT** only if (a) the document **fits** in the window **and** (b) the query requires **global, cross-section synthesis** that retrieval would likely miss.
- Prefer **RAG** when the document **does not fit** the window, or when **high-precision snippet retrieval** is likely effective (e.g., names, dates, localized facts), or when **efficiency** is a concern and quality is similar.

⚠️ Make sure that `<decision>` is fully supported by your `<reflection>` and follows the Decision Rules.

---

## Output Format

<thinking>
  <step1>...</step1>
  <step2>...</step2>
  <step3>...</step3>
  <step4>...</step4>
  <step5>...</step5>
  <step6_efficiency>...</step6_efficiency>
  <reflection>...</reflection>
  <decision>...</decision>
</thinking>
"""

def build_rag_chunks_info(rag_chunks):
    """
    构建 RAG 块的信息字符串和元信息

    :param rag_chunks: RAG 块列表
    :return: RAG 块信息字符串和元信息字符串
    """
    if rag_chunks:
        rag_chunks_str = ""
        for i, c in enumerate(rag_chunks):
            chunk_title = c['chunk']
            chunk_text = c['text'].strip()
            rag_chunks_str += f"\n{chunk_title}:\n{chunk_text}"
        rag_chunks_meta = f"- num_chunks: {len(rag_chunks)}"
    else:
        rag_chunks_str = "(No RAG chunks retrieved)"
        rag_chunks_meta = "- num_chunks: 0"
    return rag_chunks_str, rag_chunks_meta
