"""
计算所有预训练数据集的总token量
根据 PRETRAINING_DATASETS 和 DATASET_PREFIX 配置，
扫描每个数据集的 token_stats.json 文件，统计总token数
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.pretrain_ds_config import PRETRAINING_DATASETS, DATASET_PREFIX


def calculate_dataset_tokens():
    """
    计算每个数据集组的总token量
    
    Returns:
        dict: {dataset_name: total_tokens}
    """
    dataset_tokens = {}
    dataset_details = {}  # 详细信息，包含每个子路径的token数
    
    print("=" * 80)
    print("开始计算数据集token量...")
    print("=" * 80)
    
    for dataset_name, paths in PRETRAINING_DATASETS.items():
        print(f"\n处理数据集组: {dataset_name}")
        total_tokens = 0
        subset_details = {}
        
        for relative_path in paths:
            # 确定数据集前缀
            # relative_path 格式可能是 "Nemotron-CC-v2.1/High-Quality" 
            # 需要找到对应的前缀
            prefix = None
            for prefix_key, prefix_path in DATASET_PREFIX.items():
                # 尝试匹配前缀
                if relative_path.startswith(prefix_key):
                    prefix = prefix_path
                    break
                # 也支持直接使用数据集名称
                elif relative_path.split('/')[0] in prefix_key or prefix_key in relative_path:
                    prefix = prefix_path
                    break
            
            # 如果没有找到明确的前缀，尝试根据路径判断
            if prefix is None:
                if 'Nemotron' in relative_path:
                    prefix = DATASET_PREFIX.get('Nemotron')
                elif 'dolma' in relative_path:
                    prefix = DATASET_PREFIX.get('dolma3')
            
            if prefix is None:
                print(f"  ⚠️  警告: 无法确定路径 {relative_path} 的前缀，跳过")
                continue
            
            # 构建完整路径
            full_path = Path(prefix) / relative_path
            
            if not full_path.exists():
                print(f"  ⚠️  警告: 路径不存在: {full_path}")
                subset_details[relative_path] = 0
                continue
            
            # 递归查找所有 token_stats.json 文件
            subset_tokens = 0
            json_files = list(full_path.rglob("token_stats.json"))
            
            print(f"  处理: {relative_path}")
            print(f"    找到 {len(json_files)} 个 token_stats.json 文件")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                        subset_tokens += stats.get('total_tokens', 0)
                except Exception as e:
                    print(f"    ⚠️  读取文件失败: {json_file}, 错误: {e}")
            
            print(f"    总token数: {subset_tokens:,}")
            subset_details[relative_path] = subset_tokens
            total_tokens += subset_tokens
        
        dataset_tokens[dataset_name] = total_tokens
        dataset_details[dataset_name] = subset_details
        print(f"  {dataset_name} 总计: {total_tokens:,} tokens")
    
    return dataset_tokens, dataset_details


def print_summary(dataset_tokens, dataset_details):
    """打印汇总信息"""
    print("\n" + "=" * 80)
    print("数据集Token量统计汇总")
    print("=" * 80)
    
    total_all_tokens = sum(dataset_tokens.values())
    
    # 按token数排序
    sorted_datasets = sorted(dataset_tokens.items(), key=lambda x: x[1], reverse=True)
    
    for dataset_name, total_tokens in sorted_datasets:
        percentage = (total_tokens / total_all_tokens * 100) if total_all_tokens > 0 else 0
        print(f"\n{dataset_name}:")
        print(f"  总token数: {total_tokens:,}")
        print(f"  占比: {percentage:.2f}%")
        
        # 打印子集详情
        if dataset_name in dataset_details:
            print(f"  子集详情:")
            for path, tokens in dataset_details[dataset_name].items():
                sub_percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
                print(f"    - {path}: {tokens:,} tokens ({sub_percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"总计: {total_all_tokens:,} tokens")
    print(f"总计: {total_all_tokens / 1e12:.2f} T tokens")
    print("=" * 80)


def save_results(dataset_tokens, dataset_details, output_file="token_statistics.json"):
    """保存结果到JSON文件"""
    output_path = Path(__file__).parent / output_file
    
    results = {
        "summary": dataset_tokens,
        "details": dataset_details,
        "total_tokens": sum(dataset_tokens.values())
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    dataset_tokens, dataset_details = calculate_dataset_tokens()
    print_summary(dataset_tokens, dataset_details)
    save_results(dataset_tokens, dataset_details)
    
    return dataset_tokens, dataset_details


if __name__ == "__main__":
    main()
