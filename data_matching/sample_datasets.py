"""
数据采样主程序
根据配置的比例和目标数据量，从各个数据集中按比例随机采样数据

主要功能：
1. 根据 total_size_in_TB 和 PRETRAINING_DATASET_RATIOS 计算每个子集需要抽取的token量
2. 读取每个数据集的 token_stats.json，获取每个parquet文件的token量
3. 根据比例计算每个parquet文件需要采样的数据量
4. 从parquet文件中根据 token_number 字段随机采样数据

注意事项：
- 每个parquet文件所在文件夹都有一个 token_stats.json 文件
- parquet文件中有 token_number 字段记录每条数据的token量
- 采样是随机的，需要累计token量直到达到目标
"""
import json
import sys
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.pretrain_ds_config import (
    PRETRAINING_DATASETS, 
    DATASET_PREFIX,
    total_size_in_TB,
    PRETRAINING_DATASET_RATIOS
)

# ========== 配置参数（请根据需要修改） ==========
RANDOM_SEED = 42  # 随机种子，用于采样的可重复性
OUTPUT_DIR = "/path/to/output"  # 采样数据输出根目录，请修改为实际路径
# ============================================


def calculate_target_tokens():
    """
    根据 total_size_in_TB 和 PRETRAINING_DATASET_RATIOS 计算每个数据集需要的token量
    
    Returns:
        dict: {dataset_name: target_tokens}
    """
    # 1TB ≈ 250B tokens (这是一个估算，可能需要根据实际情况调整)
    # TODO: 请确认这个换算比例是否正确
    tokens_per_tb = 250_000_000_000  # 250B tokens per TB
    
    total_target_tokens = total_size_in_TB * tokens_per_tb
    
    target_tokens = {}
    for dataset_name, ratio in PRETRAINING_DATASET_RATIOS.items():
        target_tokens[dataset_name] = int(total_target_tokens * ratio)
    
    print("=" * 80)
    print("目标采样量计算")
    print("=" * 80)
    print(f"目标总量: {total_size_in_TB} TB = {total_target_tokens:,} tokens")
    print(f"总量: {total_target_tokens / 1e9:.2f}B tokens")
    print("\n各数据集目标token量:")
    for dataset_name, tokens in target_tokens.items():
        print(f"  {dataset_name}: {tokens:,} tokens ({tokens/1e9:.2f}B, {PRETRAINING_DATASET_RATIOS[dataset_name]*100:.1f}%)")
    print("=" * 80)
    
    return target_tokens


def get_dataset_actual_tokens(dataset_name, paths):
    """
    计算数据集的实际总token量
    
    Args:
        dataset_name: 数据集名称
        paths: 数据集路径列表
    
    Returns:
        dict: {
            'total_tokens': int,
            'subsets': {
                subset_path: {
                    'total_tokens': int,
                    'folders': {
                        folder_path: {
                            'total_tokens': int,
                            'files': {filename: token_count}
                        }
                    }
                }
            }
        }
    """
    dataset_info = {
        'total_tokens': 0,
        'subsets': {}
    }
    
    for relative_path in paths:
        # 确定数据集前缀
        prefix = None
        for prefix_key, prefix_path in DATASET_PREFIX.items():
            if relative_path.startswith(prefix_key) or prefix_key in relative_path or relative_path.split('/')[0] in prefix_key:
                prefix = prefix_path
                break
        
        if prefix is None:
            if 'Nemotron' in relative_path:
                prefix = DATASET_PREFIX.get('Nemotron')
            elif 'dolma' in relative_path:
                prefix = DATASET_PREFIX.get('dolma3')
        
        if prefix is None:
            print(f"  ⚠️  警告: 无法确定路径 {relative_path} 的前缀，跳过")
            continue
        
        full_path = Path(prefix) / relative_path
        
        if not full_path.exists():
            print(f"  ⚠️  警告: 路径不存在: {full_path}")
            continue
        
        subset_info = {
            'total_tokens': 0,
            'folders': {}
        }
        
        # 递归查找所有 token_stats.json 文件
        json_files = list(full_path.rglob("token_stats.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    folder_path = json_file.parent
                    
                    folder_info = {
                        'total_tokens': stats.get('total_tokens', 0),
                        'files': stats.get('files', {})
                    }
                    
                    subset_info['folders'][str(folder_path)] = folder_info
                    subset_info['total_tokens'] += folder_info['total_tokens']
            except Exception as e:
                print(f"  ⚠️  读取文件失败: {json_file}, 错误: {e}")
        
        dataset_info['subsets'][relative_path] = subset_info
        dataset_info['total_tokens'] += subset_info['total_tokens']
    
    return dataset_info


def calculate_sampling_plan(target_tokens, dataset_info):
    """
    计算采样计划：为每个parquet文件分配需要采样的token量
    
    Args:
        target_tokens: 目标token量
        dataset_info: 数据集信息（包含实际token量）
    
    Returns:
        dict: {
            folder_path: {
                parquet_file: {
                    'total_tokens': int,  # 该文件的总token量
                    'sample_tokens': int   # 需要采样的token量
                }
            }
        }
    """
    actual_tokens = dataset_info['total_tokens']
    
    if actual_tokens == 0:
        print("  ⚠️  警告: 数据集实际token量为0，无法采样")
        return {}
    
    # 计算采样比例
    sampling_ratio = min(target_tokens / actual_tokens, 1.0)
    
    print(f"  实际token量: {actual_tokens:,}")
    print(f"  目标token量: {target_tokens:,}")
    print(f"  采样比例: {sampling_ratio:.4f} ({sampling_ratio*100:.2f}%)")
    
    sampling_plan = {}
    
    # 为每个文件计算需要采样的token量
    for subset_path, subset_info in dataset_info['subsets'].items():
        for folder_path, folder_info in subset_info['folders'].items():
            folder_plan = {}
            
            for filename, file_tokens in folder_info['files'].items():
                sample_tokens = int(file_tokens * sampling_ratio)
                
                if sample_tokens > 0:
                    folder_plan[filename] = {
                        'total_tokens': file_tokens,
                        'sample_tokens': sample_tokens
                    }
            
            if folder_plan:
                sampling_plan[folder_path] = folder_plan
    
    return sampling_plan


def sample_from_parquet(parquet_path, target_tokens, random_state):
    """
    从parquet文件中随机采样数据，直到达到目标token量
    
    Args:
        parquet_path: parquet文件路径
        target_tokens: 目标token量
        random_state: 随机数生成器
    
    Returns:
        pd.DataFrame: 采样后的数据
    """
    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        
        if 'token_number' not in df.columns:
            print(f"    ⚠️  警告: {parquet_path} 缺少 token_number 字段")
            return pd.DataFrame()
        
        # 随机打乱数据
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # 累计采样直到达到目标token量
        cumsum_tokens = df['token_number'].cumsum()
        sampled_indices = cumsum_tokens <= target_tokens
        
        # 如果没有任何数据符合条件，至少取第一条
        if not sampled_indices.any():
            sampled_df = df.iloc[:1]
        else:
            # 找到最后一个满足条件的索引
            last_idx = sampled_indices.sum()
            sampled_df = df.iloc[:last_idx]
        
        actual_tokens = sampled_df['token_number'].sum()
        
        return sampled_df, actual_tokens
    
    except Exception as e:
        print(f"    ⚠️  采样失败: {parquet_path}, 错误: {e}")
        return pd.DataFrame(), 0


def sample_dataset(dataset_name, target_tokens, dataset_info, sampling_plan, output_dir):
    """
    对单个数据集进行采样，保持原始目录结构
    
    Args:
        dataset_name: 数据集名称
        target_tokens: 目标token量
        dataset_info: 数据集信息
        sampling_plan: 采样计划
        output_dir: 输出根目录
    """
    print(f"\n处理数据集: {dataset_name}")
    
    if not sampling_plan:
        print(f"  跳过: 没有采样计划")
        return
    
    # 初始化随机数生成器
    rng = random.Random(RANDOM_SEED)
    
    total_sampled_tokens = 0
    total_sampled_rows = 0
    sampled_files_count = 0
    file_counter = 0  # 用于为每个文件生成确定性的种子
    
    # 按文件夹处理，保持原始目录结构
    for folder_path, folder_plan in tqdm(sampling_plan.items(), desc=f"采样 {dataset_name}"):
        folder_path_obj = Path(folder_path)
        
        for filename, file_plan in folder_plan.items():
            parquet_path = folder_path_obj / filename
            
            if not parquet_path.exists():
                print(f"    ⚠️  文件不存在: {parquet_path}")
                continue
            
            # 采样 - 使用确定性的种子（基于全局种子和文件计数）
            file_seed = RANDOM_SEED + file_counter
            file_counter += 1
            
            sampled_df, actual_tokens = sample_from_parquet(
                parquet_path,
                file_plan['sample_tokens'],
                file_seed
            )
            
            if not sampled_df.empty:
                # 计算相对于某个数据集根目录的相对路径
                # 需要找到数据集前缀来计算相对路径
                relative_folder = None
                for prefix_key, prefix_path in DATASET_PREFIX.items():
                    prefix_path_obj = Path(prefix_path)
                    try:
                        relative_folder = folder_path_obj.relative_to(prefix_path_obj)
                        break
                    except ValueError:
                        continue
                
                if relative_folder is None:
                    # 如果无法确定相对路径，使用文件夹名称
                    relative_folder = folder_path_obj.name
                
                # 创建输出目录，保持原始目录结构
                output_folder = Path(output_dir) / relative_folder
                output_folder.mkdir(parents=True, exist_ok=True)
                
                # 保存采样后的文件，保持原文件名
                output_file = output_folder / filename
                sampled_df.to_parquet(output_file, index=False)
                
                total_sampled_tokens += actual_tokens
                total_sampled_rows += len(sampled_df)
                sampled_files_count += 1
    
    # 保存数据集级别的采样统计
    if sampled_files_count > 0:
        print(f"  ✓ 完成采样")
        print(f"    目标token量: {target_tokens:,}")
        print(f"    实际采样token量: {total_sampled_tokens:,}")
        print(f"    采样行数: {total_sampled_rows:,}")
        print(f"    采样文件数: {sampled_files_count}")
        print(f"    完成度: {total_sampled_tokens/target_tokens*100:.2f}%")
        
        # 保存采样统计到输出根目录
        stats = {
            'dataset_name': dataset_name,
            'target_tokens': target_tokens,
            'actual_sampled_tokens': int(total_sampled_tokens),
            'sampled_rows': total_sampled_rows,
            'sampled_files': sampled_files_count,
            'completion_rate': total_sampled_tokens / target_tokens if target_tokens > 0 else 0,
            'random_seed': RANDOM_SEED
        }
        
        stats_file = Path(output_dir) / f"{dataset_name}_sampling_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"    统计文件: {stats_file}")
    else:
        print(f"  ⚠️  没有采样到任何数据")


def main():
    """主函数"""
    print("=" * 80)
    print("数据采样程序")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  随机种子: {RANDOM_SEED}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  目标数据量: {total_size_in_TB} TB")
    print("=" * 80)
    
    # 检查输出目录配置
    if OUTPUT_DIR == "/path/to/output":
        print("\n⚠️  错误: 请先配置 OUTPUT_DIR 参数!")
        print("   请在文件顶部修改 OUTPUT_DIR 为实际的输出路径")
        return
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    
    # 1. 计算目标token量
    target_tokens_dict = calculate_target_tokens()
    
    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 对每个数据集进行采样
    for dataset_name, target_tokens in target_tokens_dict.items():
        if dataset_name not in PRETRAINING_DATASETS:
            print(f"\n⚠️  警告: 数据集 {dataset_name} 在 PRETRAINING_DATASETS 中未定义，跳过")
            continue
        
        paths = PRETRAINING_DATASETS[dataset_name]
        
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*80}")
        
        # 2.1 获取数据集实际token量和文件信息
        print("  扫描数据集...")
        dataset_info = get_dataset_actual_tokens(dataset_name, paths)
        
        # 2.2 计算采样计划
        print("  计算采样计划...")
        sampling_plan = calculate_sampling_plan(target_tokens, dataset_info)
        
        # 2.3 执行采样
        sample_dataset(dataset_name, target_tokens, dataset_info, sampling_plan, output_path)
    
    print("\n" + "=" * 80)
    print("所有数据集采样完成!")
    print(f"输出目录: {output_path}")
    print(f"采样使用的随机种子: {RANDOM_SEED}")
    print("=" * 80)


if __name__ == "__main__":
    main()
