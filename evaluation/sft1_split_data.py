

import json
import random
import os
import hashlib
from typing import List, Dict
from collections import defaultdict

from eval_utils import generate_id

# =========================================================================
# ====== 辅助函数 (Helper Functions) ======
# =========================================================================

def add_ids_to_data(data: List[Dict]) -> List[Dict]:
    """为数据集中的每个样本添加_id"""
    for example in data:
        # 检查example是否是嵌套结构
        if "example" in example:
            item_to_hash = example.get("example", example)
            item_to_hash["_id"] = generate_id(item_to_hash)
        else:
            item_to_hash = example
            item_to_hash["_id"] = generate_id(item_to_hash)
    return data

# =========================================================================
# ====== 路径与参数配置 (Path and Parameter Configuration) ======
# =========================================================================

# Input: routing-record files produced by 3router_judge_sample_bon.py (steps 1-3 of the README).
input_files = [
    # Answer Model: qwen3-30b-a3b-instruct-2507 (produced by the demo chain in the README)
    "evaluation/prediction/qwen3-30b-a3b-instruct-2507/flattened_merged_predictions_qwen3-235b-a22b-instruct-2507_comparison_routerqwen3-235b-a22b-instruct-2507_preroute_bon1_thinkingFalse.json",
]

# 输出目录配置
tag = "from_multi_answer_models"
# 数据集划分比例：训练:验证:测试 = 7:1:2
train_ratio = 0.7
eval_ratio = 0.1
test_ratio = 0.2
random_seed = 42
split_dir = f"distillation_datasets/{tag}_{train_ratio}_{eval_ratio}_{test_ratio}_{random_seed}"
os.makedirs(split_dir, exist_ok=True)

config = {
    "tag": tag,
    "train_ratio": train_ratio,
    "eval_ratio": eval_ratio,
    "test_ratio": test_ratio,
    "random_seed": random_seed,
    "input_files": input_files,
}

config_file = os.path.join(split_dir, f"config.json")
split_idx_file = os.path.join(split_dir, f"global_split_indices.json")

# 最终输出文件路径
train_file = os.path.join(split_dir, f"train_data_sft.json")
eval_file = os.path.join(split_dir, f"eval_data_sft.json")
test_file = os.path.join(split_dir, f"test_data_sft.json")

if not os.path.exists(eval_file):
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    do_update = True
else:
    with open(config_file, "r", encoding="utf-8") as f:
        read_config = json.load(f)

    # check config == read_config 
    if config.keys() != read_config.keys():
        do_update = True
    else:
        do_update = False
        for k in config.keys():
            if config[k] != read_config[k]:
                do_update = True
                break
    if do_update:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

if do_update:
    print("待更新更新")
    # 全局划分索引文件的路径

    # =========================================================================
    # ====== 1. 全局Q&A对划分 (Global Q&A Split) ======
    # =========================================================================
    print("步骤 1: 进行全局Q&A对划分...")

    if not os.path.exists(split_idx_file):
        print(f"未找到全局划分文件，正在创建: {split_idx_file}")
        
        all_unique_qas = {} # 使用字典来存储唯一的QA ID及其元信息
        
        # 遍历所有输入文件以构建一个完整的、唯一的Q&A对集合
        print("扫描所有输入文件以构建全局问题列表...")
        for file_path in input_files:
            print(f"  - 读取文件: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            data = add_ids_to_data(data)
            for item in data:
                example = item.get("example", item)
                qa_id = example["_id"]
                if qa_id not in all_unique_qas:
                    all_unique_qas[qa_id] = {
                        "type": example.get("type", example.get("type", "unknown")),
                        "level": example.get("level", example.get("level", "unknown"))
                    }

        print(f"发现 {len(all_unique_qas)} 个独立的Q&A对。")

        # 按 type 和 level 的复合键对所有唯一的Q&A对进行分组
        qa_by_composite_key = defaultdict(list)
        for qa_id, meta in all_unique_qas.items():
            composite_key = f"{meta['type']} X {meta['level']}"
            qa_by_composite_key[composite_key].append(qa_id)

        # 进行分层随机划分
        random.seed(random_seed)
        train_ids = []
        eval_ids = []
        test_ids = []

        print("按 type x level 复合键进行分层采样...")
        sorted_keys = sorted(qa_by_composite_key.keys()) # 排序以保证每次运行输出顺序一致
        for key in sorted_keys:
            ids = qa_by_composite_key[key]
            random.shuffle(ids)
            
            n_total = len(ids)
            if n_total == 0:
                continue
                
            # 计算划分点
            n_train = max(1, int(n_total * train_ratio))  # 确保至少1个样本在训练集
            n_eval = max(1, int(n_total * eval_ratio))    # 确保至少1个样本在验证集
            
            # 处理小数据集情况
            if n_total <= 3:
                # 如果总数很小，按7:1:2比例分配，但确保每部分至少1个
                n_train = max(1, int(n_total * 0.7))
                n_eval = max(1, int(n_total * 0.1))
                n_test = max(1, n_total - n_train - n_eval)
                
                # 调整确保总和正确
                if n_train + n_eval + n_test > n_total:
                    n_train = n_total - n_eval - n_test
            else:
                n_test = n_total - n_train - n_eval
                # 确保测试集至少有1个样本
                if n_test < 1:
                    n_test = 1
                    n_eval = max(1, (n_total - n_train - n_test) // 2)
                    n_train = n_total - n_eval - n_test

            train_ids.extend(ids[:n_train])
            eval_ids.extend(ids[n_train:n_train+n_eval])
            test_ids.extend(ids[n_train+n_eval:])
            
            print(f"  - 复合键 '{key}': {n_total}个 -> 训练集 {n_train}个, 验证集 {n_eval}个, 测试集 {n_test}个")

        # 保存全局划分的ID集合
        with open(split_idx_file, "w", encoding="utf-8") as f:
            json.dump({"train_ids": train_ids, "eval_ids": eval_ids, "test_ids": test_ids}, f)
        
        print(f"全局划分完成，总计: 训练集 {len(train_ids)} 个Q&A, 验证集 {len(eval_ids)} 个Q&A, 测试集 {len(test_ids)} 个Q&A")
    else:
        print(f"加载已存在的全局划分文件: {split_idx_file}")
        with open(split_idx_file, "r", encoding="utf-8") as f:
            split_indices = json.load(f)
        train_ids = split_indices["train_ids"]
        eval_ids = split_indices["eval_ids"]
        test_ids = split_indices["test_ids"]

    # 将列表转换为集合以提高查找效率
    train_id_set = set(train_ids)
    eval_id_set = set(eval_ids)
    test_id_set = set(test_ids)

    # =========================================================================
    # ====== 2. 合并、归类与筛选 (Merge, Classify, and Filter) ======
    # =========================================================================
    print("\n步骤 2: 合并、归类与筛选多个实验文件...")

    all_train_data = []
    all_eval_data = []
    all_test_data = []

    for file_path in input_files:
        print(f"  - 正在处理文件: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data = add_ids_to_data(data)
        
        for item in data:
            example = item.get("example", item)
            qa_id = example["_id"]
            item["source"] = file_path
            if qa_id in train_id_set:
                all_train_data.append(item)
            elif qa_id in eval_id_set:
                all_eval_data.append(item)
            elif qa_id in test_id_set:
                all_test_data.append(item)

    print(f"合并完成，总计: 训练池 {len(all_train_data)} 条记录, 验证池 {len(all_eval_data)} 条记录, 测试池 {len(all_test_data)} 条记录")

    # 对训练数据进行拒绝采样/筛选
    print("\n步骤 3: 对训练数据进行拒绝采样...")
    original_train_count = len(all_train_data)
    filtered_train_data = [
        item for item in all_train_data if item.get("is_correct") is True
    ]

    print("\n步骤 4: 统计筛选后训练集的类别分布...")
    filtered_train_distribution = defaultdict(int)
    for item in filtered_train_data:
        example = item.get("example", item)
        composite_key = f"{example.get('type', 'unknown')}_{example.get('level', 'unknown')}"
        filtered_train_distribution[composite_key] += 1

    print("筛选后训练集各类别样本数:")
    sorted_filtered_keys = sorted(filtered_train_distribution.keys())
    for key in sorted_filtered_keys:
        print(f"  - 复合键 '{key}': {filtered_train_distribution[key]}个")

    # =========================================================================
    # ====== 3. 保存最终结果 (Save Final Datasets) ======
    # =========================================================================
    print("\n步骤 5: 保存最终数据集...")

    # 保存筛选后的训练集
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(filtered_train_data, f, ensure_ascii=False, indent=2)

    # 保存未经过滤的验证集
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_data, f, ensure_ascii=False, indent=2)

    # 保存未经过滤的测试集
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(all_test_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*30)
    print("✅ 数据集构造完成！")
    print(f"原始训练池样本数: {original_train_count}")
    print(f"筛选后最终训练集样本数: {len(filtered_train_data)} (只保留 is_correct: true)")
    print(f"最终验证集样本数: {len(all_eval_data)} (保持不变)")
    print(f"最终测试集样本数: {len(all_test_data)} (保持不变)")
    print(f"训练集保存至: {train_file}")
    print(f"验证集保存至: {eval_file}")
    print(f"测试集保存至: {test_file}")
    print("="*30)