import argparse
import os
import json
import pickle
import time
from tqdm import tqdm
from eval_utils import (
    load_jsonl,
    create_msgs,
    dump_jsonl,
    call_model,
    get_tokenizer
)
import tiktoken
import concurrent.futures
import traceback

def process_example(eg, rag_chunk_cache, context_length, context_type, tokenizer, eval_model, args):
    file_name = eg['file']
    key = (file_name, eg['question'])

    # 尝试从pkl缓存读取
    if key in rag_chunk_cache:
        eg["context"] = rag_chunk_cache[key]
    else:
        # live retrieval fallback (requires torch, llama_index and the search/ package)
        import torch  # noqa: F401
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from transformers import logging as hf_logging
        from search.simpleHybridSearcher import SimpleHybridSearcher
        hf_logging.set_verbosity_error()
        eg_file_path = f"./datasets/{context_length}/{context_type}/{file_name}"
        with open(eg_file_path, 'r', encoding='utf-8') as f:
            eg_txt = f.read()

        transformations = []
        splitter = SentenceSplitter(
            include_metadata=True, include_prev_next_rel=True,
            chunk_size=600,
            chunk_overlap=100,
            separator=' ',
            paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?')
        transformations.append(splitter)

        emb_model = "./embedding_models/gte-multilingual-base"
        rerank_model = "./embedding_models/gte-multilingual-reranker-base"
        embed_model = HuggingFaceEmbedding(model_name=emb_model, trust_remote_code=True)
        transformations.append(embed_model)
        pipeline = IngestionPipeline(transformations=transformations)

        eg_doc = Document(text=eg_txt)
        documents = [eg_doc]
        nodes = pipeline.run(documents=documents, show_progress=False)
        config = {
            "class_name": "SimpleHybridSearcher",
            "class_file": "simpleHybridSearcher",
            "remove_if_exists": False,
            "thread_num": 1,
            "rerank_size": 5,
            "vector_ratio": 0.5,
            "embed_model_name": emb_model,
            "rerank_model": rerank_model,
        }

        searcher = SimpleHybridSearcher(config, nodes)
        search_nodes = searcher.process(eg['question'])
        eg["context"] = ''
        for i, node in enumerate(search_nodes):
            node_txt = node.text
            eg["context"] += f'chunk {i}: {node_txt}.' + '\n\n'

    msgs, prompt = create_msgs(
        tokenizer, eg, data_name=context_type, model_name="rag", eval_model=eval_model, thinking=False, reserved_ratio=0.95
    )
    try:
        response = call_model(msgs, eval_model, thinking=args.thinking)
        # print("response:", response)
        return response, eg
    except Exception as e:
        print("ERROR:", e)
        print(traceback.format_exc())
        print("ERR RAG")
        print(context_length, context_type, tokenizer, eval_model)
        time.sleep(10)
        return False, eg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default="hallu", help='the type of the query')
    parser.add_argument('--context_type', type=str, default="financial", help='the type of the context')
    parser.add_argument('--context_length', type=str, default="32k", help='the length of the context')
    parser.add_argument('--eval_model', type=str, default="qwen3-30b-a3b-instruct-2507", help='model')
    parser.add_argument('--ragchunk_pkl', type=str, default=None, help='pkl file with rag chunk cache')
    parser.add_argument('--thinking', action='store_true', default=False)
    parser.add_argument('--rewrite', action='store_true', default=False)
    parser.add_argument('--limit', type=int, default=None, help='process only the first N queries (low-cost trial)')
    args = parser.parse_args()
    print(f"===processing：===\n")
    print(f"eval_model: {args.eval_model}")
    print(f"context_length: {args.context_length}")
    print(f"context_type: {args.context_type}")
    print(f"query_type: {args.query_type}")
    print(f"thinking: {args.thinking}")
    print(f"rewrite: {args.rewrite}")

    eval_model = args.eval_model
    query_type = args.query_type
    context_type = args.context_type
    context_length = args.context_length
    ragchunk_pkl = args.ragchunk_pkl or f"./datasets/all_query_map_ragchunks.pkl"

    if os.path.exists(ragchunk_pkl):
        print(f"加载RAG缓存: {ragchunk_pkl}")
        with open(ragchunk_pkl, "rb") as f:
            rag_chunk_cache = pickle.load(f)
        print(f"RAG缓存条目数: {len(rag_chunk_cache)}")
    else:
        print(f"未找到RAG缓存 {ragchunk_pkl}，后续全部实时retrieval！")
        rag_chunk_cache = {}

    data_path = f'./datasets/query/{context_length}_{context_type}_{query_type}.jsonl'
    examples = load_jsonl(data_path)
    if args.limit:
        examples = examples[:args.limit]
        print(f"limit: processing first {len(examples)} queries")

    output_dir = f"./evaluation/prediction/{eval_model}"

    output_path = f'{output_dir}/rag_preds_{eval_model}_{context_length}_{context_type}_{query_type}{"_thinking" if args.thinking else ""}.jsonl'
    print(f"output_path: {output_path}")

    if os.path.exists(output_path) and not args.rewrite:
        print(f"已存在预测结果 {output_path}，跳过！")
        exit(0)

    os.makedirs(output_dir, exist_ok=True)
    print(f"make dirs: {output_dir}")

    tokenizer = get_tokenizer(eval_model)

    preds = []

    if eval_model in ['qwen3-235b-a22b-thinking-2507', 'qwen3-235b-a22b-instruct-2507']:
        max_workers = 20
    elif eval_model in ['deepseek-r1-0528', "qwen-max-latest"]:
        max_workers = 7
    else:
    # 保持原有的默认逻辑
        if 'qwen' in eval_model:
            max_workers = 30
        else:
            max_workers = 16
    if context_length == "32k":
        max_workers *= 1.5
        max_workers = int(max_workers)
    print(f"true max_workers: {max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_example,
                example,
                rag_chunk_cache,
                context_length,
                context_type,
                tokenizer,
                eval_model,
                args
            )
            for example in examples
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result, eg = future.result()
            if query_type in ['location', 'reasoning', 'hallu']:
                preds.append({
                        "type": eg['type'],
                        "level": eg['level'],
                        "file": eg['file'],
                        "context_order": eg.get('context_order', None),
                        "question": eg['question'],
                        "reason": result[0],
                        "prediction": result[1],
                        "ground_truth": eg['answer'],
                    }
                )
            if query_type in ['comp']:
                preds.append(
                    {
                        "type": eg['type'],
                        "level": eg['level'],
                        "file": eg['file'],
                        "comp_parts": eg.get('comp_parts', None),
                        "question": eg['question'],
                        "reason": result[0],
                        "prediction": result[1],
                        "ground_truth": eg['answer'],
                    }
                )
    dump_jsonl(preds, output_path)
    print(f"rag_preds_{eval_model}_{context_length}_{context_type}_{query_type} saving to {output_path}")