import argparse
import os
import json
import pickle
from tqdm import tqdm
import torch
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from search.simpleHybridSearcher import SimpleHybridSearcher

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import logging
logging.set_verbosity_error()

splitter = SentenceSplitter(
    include_metadata=True,
    include_prev_next_rel=True,
    chunk_size=600,
    chunk_overlap=100,
    separator=' ',
    paragraph_separator='\n\n\n',
    secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'
)
embed_model_path = "./embedding_models/gte-multilingual-base"
rerank_model_path = "./embedding_models/gte-multilingual-reranker-base"

def find_doc_path(base_dir, length, domain, file):
    return os.path.join(base_dir, length, domain, file)

def load_all_queries(query_dir):
    queries = []
    for fname in sorted(os.listdir(query_dir)):
        if fname.endswith('.jsonl'):
            path = os.path.join(query_dir, fname)
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    q = json.loads(line)
                    # 推断 domain 字段
                    if 'domain' not in q:
                        fn = q['file']
                        if fn.endswith('.md'):
                            q['domain'] = 'paper'
                        elif fn.endswith('.txt'):
                            if 'financial' in fname:
                                q['domain'] = 'financial'
                            elif 'book' in fname:
                                q['domain'] = 'book'
                            else:
                                q['domain'] = 'book'
                    # 推断 length 字段
                    if 'length' not in q:
                        if fname.startswith('32k'):
                            q['length'] = '32k'
                        elif fname.startswith('128k'):
                            q['length'] = '128k'
                    queries.append(q)
    return queries

def process_example_rag_q(base_dir, eg):
    file_path = find_doc_path(base_dir, eg["length"], eg["domain"], eg["file"])
    with open(file_path, 'r', encoding='utf-8') as f:
        eg_txt = f.read()
    embed_model = HuggingFaceEmbedding(model_name=embed_model_path, trust_remote_code=True)
    pipeline = IngestionPipeline(transformations=[splitter, embed_model])
    nodes = pipeline.run(documents=[Document(text=eg_txt)], show_progress=False)
    searcher = SimpleHybridSearcher({
        "class_name": "SimpleHybridSearcher",
        "class_file": "simpleHybridSearcher",
        "remove_if_exists": False,
        "thread_num": 1,
        "rerank_size": 5,
        "vector_ratio": 0.5,
        "embed_model_name": embed_model_path,
        "rerank_model": rerank_model_path,
    }, nodes)
    search_nodes = searcher.process(eg['question'])
    eg["rag_context"] = '\n\n'.join([f"chunk {i}: {node.text}" for i, node in enumerate(search_nodes)])
    del embed_model, pipeline, searcher, search_nodes
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return eg

def process_example(base_dir, eg):
    eg = process_example_rag_q(base_dir, eg.copy())
    torch.cuda.empty_cache()
    return eg

def task_runner(base_dir, chunk, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch, gc
    results = {}
    for eg in tqdm(chunk, total=len(chunk), desc=f"Processing GPU {gpu_id}"):
        try:
            out = process_example(base_dir, eg)
            results[(out["file"], out["question"])] = out["rag_context"]
        except Exception as e:
            print(f"[ERROR][GPU {gpu_id}] Failed on {eg['file']}: {e}")
            raise e
    torch.cuda.empty_cache()
    gc.collect()
    return results

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Path to the dataset folder.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Processes per GPU")
    args = parser.parse_args()

    query_dir = os.path.join(args.dataset_dir, "query")
    if not os.path.exists(query_dir):
        raise FileNotFoundError(f"{query_dir} does not exist!")

    print(f"🔍 Loading all queries from {query_dir} ...")
    all_examples = load_all_queries(query_dir)
    print("QUERIES LOADED:", len(all_examples))

    output_path = os.path.join(args.dataset_dir, "all_query_map_ragchunks.pkl")
    already_done = {}
    if os.path.exists(output_path):
        print(f"🟡 检测到已有结果 {output_path}，正在加载以跳过已完成部分 ...")
        with open(output_path, "rb") as f:
            already_done = pickle.load(f)
        print(f"已完成数量: {len(already_done)}")

    # 只处理未完成的
    to_process = []
    for eg in all_examples:
        k = (eg["file"], eg["question"])
        if k not in already_done:
            to_process.append(eg)
    print(f"本次需要补充处理: {len(to_process)}")

    if to_process:
        gpu_list = [int(x) for x in args.gpus.split(",")]
        num_gpus = len(gpu_list)
        total_workers = num_gpus * args.workers_per_gpu

        chunk_size = (len(to_process) + total_workers - 1) // total_workers
        task_chunks = [
            (to_process[i * chunk_size:(i + 1) * chunk_size], gpu_list[i % num_gpus])
            for i in range(total_workers)
        ]
        print(f"🚀 Starting with {num_gpus} GPUs × {args.workers_per_gpu} workers each = {total_workers} total processes")

        all_results = {}
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            futures = [
                executor.submit(task_runner, args.dataset_dir, chunk, gpu_id)
                for chunk, gpu_id in task_chunks if chunk  # 避免空chunk
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                try:
                    res = future.result()
                    all_results.update(res)
                except Exception as e:
                    print("[FATAL ERROR]:", e)

        # 合并旧结果与新结果
        already_done.update(all_results)
        with open(output_path, "wb") as f:
            pickle.dump(already_done, f)
        print(f"✅ 本轮处理结束，所有结果已写入：{output_path}")
    else:
        print("🎉 没有新的任务需要处理，一切已完成！")