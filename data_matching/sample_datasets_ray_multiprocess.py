"""
数据采样主程序 - Ray分布式并行版本
使用 Ray 分布式框架在多个节点上并行采样数据
每个节点内部使用多进程处理分配给它的数据文件

主要功能：
1. 根据 total_size_in_TB 和 PRETRAINING_DATASET_RATIOS 计算每个子集需要抽取的token量
2. 读取每个数据集的 token_stats.json，获取每个parquet文件的token量
3. 根据比例计算每个parquet文件需要采样的数据量
4. 使用Ray+多进程并行从parquet文件中随机采样数据
"""
import gc
import ray
import json
import sys
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import socket
import logging
from multiprocessing import Pool

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.pretrain_ds_config import (
    PRETRAINING_DATASETS, 
    DATASET_PREFIX,
    total_size_in_TB,
    PRETRAINING_DATASET_RATIOS
)

# 导入工作函数模块
from data_matching.sample_single_file import sample_single_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== 配置参数（请根据需要修改） ==========
RANDOM_SEED = 42  # 随机种子，用于采样的可重复性
OUTPUT_DIR = "/path/to/output"  # 采样数据输出根目录，请修改为实际路径
NUM_WORKERS_PER_NODE = 10  # 每个节点的并行进程数
EXCLUDE_IPS = []  # 需要排除的节点IP列表
# ============================================


def calculate_target_tokens():
    """
    根据 total_size_in_TB 和 PRETRAINING_DATASET_RATIOS 计算每个数据集需要的token量
    
    Returns:
        dict: {dataset_name: target_tokens}
    """
    tokens_per_tb = 250_000_000_000  # 250B tokens per TB
    total_target_tokens = total_size_in_TB * tokens_per_tb
    
    target_tokens = {}
    for dataset_name, ratio in PRETRAINING_DATASET_RATIOS.items():
        target_tokens[dataset_name] = int(total_target_tokens * ratio)
    
    logger.info("=" * 80)
    logger.info("目标采样量计算")
    logger.info("=" * 80)
    logger.info(f"目标总量: {total_size_in_TB} TB = {total_target_tokens:,} tokens")
    logger.info(f"总量: {total_target_tokens / 1e9:.2f}B tokens")
    logger.info("\n各数据集目标token量:")
    for dataset_name, tokens in target_tokens.items():
        logger.info(f"  {dataset_name}: {tokens:,} tokens ({tokens/1e9:.2f}B, {PRETRAINING_DATASET_RATIOS[dataset_name]*100:.1f}%)")
    logger.info("=" * 80)
    
    return target_tokens


def get_dataset_actual_tokens(dataset_name, paths):
    """计算数据集的实际总token量并收集所有需要采样的文件信息"""
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
            logger.warning(f"  ⚠️  警告: 无法确定路径 {relative_path} 的前缀，跳过")
            continue
        
        full_path = Path(prefix) / relative_path
        
        if not full_path.exists():
            logger.warning(f"  ⚠️  警告: 路径不存在: {full_path}")
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
                logger.error(f"  ⚠️  读取文件失败: {json_file}, 错误: {e}")
        
        dataset_info['subsets'][relative_path] = subset_info
        dataset_info['total_tokens'] += subset_info['total_tokens']
    
    return dataset_info


def prepare_sampling_tasks(dataset_name, target_tokens, dataset_info):
    """
    准备采样任务列表
    
    Returns:
        list: [(parquet_path, target_tokens, output_path, file_seed), ...]
    """
    actual_tokens = dataset_info['total_tokens']
    
    if actual_tokens == 0:
        logger.warning("  ⚠️  警告: 数据集实际token量为0，无法采样")
        return []
    
    # 计算采样比例
    sampling_ratio = min(target_tokens / actual_tokens, 1.0)
    
    logger.info(f"  实际token量: {actual_tokens:,}")
    logger.info(f"  目标token量: {target_tokens:,}")
    logger.info(f"  采样比例: {sampling_ratio:.4f} ({sampling_ratio*100:.2f}%)")
    
    tasks = []
    file_counter = 0
    
    # 为每个文件创建采样任务
    for subset_path, subset_info in dataset_info['subsets'].items():
        for folder_path, folder_info in subset_info['folders'].items():
            folder_path_obj = Path(folder_path)
            
            for filename, file_tokens in folder_info['files'].items():
                sample_tokens = int(file_tokens * sampling_ratio)
                
                if sample_tokens > 0:
                    parquet_path = folder_path_obj / filename
                    
                    if not parquet_path.exists():
                        continue
                    
                    # 计算相对路径
                    relative_folder = None
                    for prefix_key, prefix_path in DATASET_PREFIX.items():
                        prefix_path_obj = Path(prefix_path)
                        try:
                            relative_folder = folder_path_obj.relative_to(prefix_path_obj)
                            break
                        except ValueError:
                            continue
                    
                    if relative_folder is None:
                        relative_folder = folder_path_obj.name
                    
                    # 输出路径
                    output_folder = Path(OUTPUT_DIR) / relative_folder
                    output_file = output_folder / filename
                    
                    # 文件种子
                    file_seed = RANDOM_SEED + file_counter
                    file_counter += 1
                    
                    tasks.append((
                        str(parquet_path),
                        sample_tokens,
                        str(output_file),
                        file_seed
                    ))
    
    logger.info(f"  准备了 {len(tasks)} 个采样任务")
    return tasks


def sample_files_on_node(task_list, num_workers):
    """
    在单个节点上使用多进程采样文件列表
    
    Args:
        task_list: 需要处理的任务列表 [(parquet_path, target_tokens, output_path, file_seed), ...]
        num_workers: 并行进程数
    
    Returns:
        处理结果统计信息
    """
    hostname = socket.gethostname()
    node_logger = logging.getLogger(f"Node-{hostname}")
    
    node_logger.info(f"节点 {hostname} 开始处理 {len(task_list)} 个文件")
    node_logger.info(f"使用 {num_workers} 个并行进程")
    
    # 准备参数列表 - 为每个任务添加worker_id
    args_list = [
        (*task, i % num_workers)
        for i, task in enumerate(task_list)
    ]
    
    # 统计信息
    total_sampled_tokens = 0
    total_sampled_rows = 0
    success_count = 0
    failed_count = 0
    
    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(sample_single_file, args_list),
            total=len(args_list),
            desc=f"[{hostname}] 采样文件"
        ))
    
    # 汇总结果
    for success, actual_tokens, row_count, output_path in results:
        if success:
            success_count += 1
            total_sampled_tokens += actual_tokens
            total_sampled_rows += row_count
        else:
            failed_count += 1
    
    # 打印总结
    node_logger.info("=" * 80)
    node_logger.info(f"节点 {hostname} 处理完成!")
    node_logger.info(f"成功处理: {success_count}/{len(task_list)} 个文件")
    node_logger.info(f"失败: {failed_count} 个文件")
    node_logger.info(f"总采样token数: {total_sampled_tokens:,}")
    node_logger.info(f"总采样行数: {total_sampled_rows:,}")
    node_logger.info("=" * 80)
    
    return {
        'hostname': hostname,
        'total_files': len(task_list),
        'success_count': success_count,
        'failed_count': failed_count,
        'total_sampled_tokens': total_sampled_tokens,
        'total_sampled_rows': total_sampled_rows
    }


@ray.remote
def sample_node_task(task_list, num_workers):
    """
    Ray远程任务：在节点上采样数据
    
    Args:
        task_list: 分配给该节点的任务列表
        num_workers: 每个节点的并行进程数
    
    Returns:
        处理结果
    """
    node_ip = ray.util.get_node_ip_address()
    hostname = socket.gethostname()
    
    try:
        result = sample_files_on_node(
            task_list=task_list,
            num_workers=num_workers
        )
        result['node_ip'] = node_ip
        result['status'] = 'success'
        return result
        
    except Exception as e:
        logger.error(f"[{hostname}] 节点处理失败: {str(e)}")
        return {
            'hostname': hostname,
            'node_ip': node_ip,
            'status': 'failed',
            'error': str(e),
            'total_files': len(task_list),
            'success_count': 0,
            'failed_count': len(task_list),
            'total_sampled_tokens': 0,
            'total_sampled_rows': 0
        }


def get_all_nodes(exclude_ips):
    """获取所有可用的Ray节点"""
    nodes = [
        node for node in ray.nodes() 
        if node['Alive'] and node.get('NodeManagerAddress', 'unknown') not in exclude_ips
    ]
    logger.info(f"发现 {len(nodes)} 个可用节点")
    return nodes


def split_tasks_to_nodes(all_tasks, all_nodes):
    """
    将任务列表均匀分配到所有节点
    
    Args:
        all_tasks: 所有需要处理的任务列表
        all_nodes: 所有可用的节点列表
    
    Returns:
        每个节点分配的任务列表
    """
    node_num = len(all_nodes)
    total_tasks = len(all_tasks)
    
    # 均匀划分
    base_size = total_tasks // node_num
    remainder = total_tasks % node_num
    
    node_task_lists = []
    start = 0
    
    for node_idx in range(node_num):
        current_size = base_size + (1 if node_idx < remainder else 0)
        end = start + current_size
        
        node_tasks = all_tasks[start:end]
        node_task_lists.append(node_tasks)
        
        logger.info(f"节点 {node_idx}: 分配 {len(node_tasks)} 个任务")
        start = end
    
    return node_task_lists


def submit_tasks_to_nodes(all_nodes, node_task_lists, num_workers_per_node):
    """
    向所有节点提交处理任务
    
    Returns:
        Ray任务引用列表
    """
    ray_tasks = []
    
    for node_idx, (node, task_list) in enumerate(zip(all_nodes, node_task_lists)):
        if not task_list:
            logger.warning(f"节点 {node_idx} 没有分配任务，跳过")
            continue
        
        # 创建指定节点上的任务
        task = sample_node_task.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], 
                soft=False
            )
        ).remote(
            task_list=task_list,
            num_workers=num_workers_per_node
        )
        
        ray_tasks.append(task)
        logger.info(f"任务已提交到节点 {node_idx} (NodeID: {node['NodeID']})")
    
    return ray_tasks


def aggregate_results(results, dataset_name, target_tokens, output_dir):
    """
    汇总所有节点的处理结果
    
    Args:
        results: 所有节点返回的结果列表
        dataset_name: 数据集名称
        target_tokens: 目标token量
        output_dir: 输出目录
    """
    logger.info("=" * 80)
    logger.info(f"数据集 {dataset_name} 处理完成，汇总结果:")
    logger.info("=" * 80)
    
    total_files = 0
    total_success = 0
    total_failed = 0
    total_sampled_tokens = 0
    total_sampled_rows = 0
    
    for idx, result in enumerate(results):
        if result['status'] == 'failed':
            logger.error(f"节点 {result['hostname']} ({result['node_ip']}) 处理失败: {result['error']}")
            continue
        
        logger.info(f"\n节点 {idx + 1}: {result['hostname']} ({result['node_ip']})")
        logger.info(f"  处理文件数: {result['total_files']}")
        logger.info(f"  成功: {result['success_count']}, 失败: {result['failed_count']}")
        logger.info(f"  采样token数: {result['total_sampled_tokens']:,}")
        logger.info(f"  采样行数: {result['total_sampled_rows']:,}")
        
        total_files += result['total_files']
        total_success += result['success_count']
        total_failed += result['failed_count']
        total_sampled_tokens += result['total_sampled_tokens']
        total_sampled_rows += result['total_sampled_rows']
    
    logger.info("\n" + "=" * 80)
    logger.info("总体统计:")
    logger.info(f"  总文件数: {total_files}")
    logger.info(f"  成功处理: {total_success}")
    logger.info(f"  处理失败: {total_failed}")
    logger.info(f"  目标token量: {target_tokens:,}")
    logger.info(f"  实际采样token数: {total_sampled_tokens:,}")
    logger.info(f"  实际采样行数: {total_sampled_rows:,}")
    logger.info(f"  完成度: {total_sampled_tokens/target_tokens*100:.2f}%")
    logger.info("=" * 80)
    
    # 保存统计信息
    stats = {
        'dataset_name': dataset_name,
        'target_tokens': target_tokens,
        'actual_sampled_tokens': total_sampled_tokens,
        'sampled_rows': total_sampled_rows,
        'sampled_files': total_success,
        'failed_files': total_failed,
        'completion_rate': total_sampled_tokens / target_tokens if target_tokens > 0 else 0,
        'random_seed': RANDOM_SEED
    }
    
    stats_file = Path(output_dir) / f"{dataset_name}_sampling_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n统计文件已保存: {stats_file}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("数据采样程序 - Ray分布式并行版本")
    logger.info("=" * 80)
    logger.info(f"配置信息:")
    logger.info(f"  随机种子: {RANDOM_SEED}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  目标数据量: {total_size_in_TB} TB")
    logger.info(f"  每节点进程数: {NUM_WORKERS_PER_NODE}")
    logger.info("=" * 80)
    
    # 检查输出目录配置
    if OUTPUT_DIR == "/path/to/output":
        logger.error("\n⚠️  错误: 请先配置 OUTPUT_DIR 参数!")
        logger.error("   请在文件顶部修改 OUTPUT_DIR 为实际的输出路径")
        return
    
    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化 Ray
    logger.info("\n连接到 Ray 集群...")
    ray.init(
        address="auto",
        runtime_env={
            "pip": ["pandas", "pyarrow"]  # 自动在所有节点安装依赖
        }
    )
    logger.info("Ray 连接成功!")
    
    try:
        # 获取所有可用节点
        all_nodes = get_all_nodes(EXCLUDE_IPS)
        if not all_nodes:
            logger.error("没有可用的节点!")
            return
        
        # 1. 计算目标token量
        target_tokens_dict = calculate_target_tokens()
        
        # 2. 对每个数据集进行采样
        for dataset_name, target_tokens in target_tokens_dict.items():
            if dataset_name not in PRETRAINING_DATASETS:
                logger.warning(f"\n⚠️  警告: 数据集 {dataset_name} 在 PRETRAINING_DATASETS 中未定义，跳过")
                continue
            
            paths = PRETRAINING_DATASETS[dataset_name]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"处理数据集: {dataset_name}")
            logger.info(f"{'='*80}")
            
            # 2.1 获取数据集实际token量和文件信息
            logger.info("  扫描数据集...")
            dataset_info = get_dataset_actual_tokens(dataset_name, paths)
            
            # 2.2 准备采样任务
            logger.info("  准备采样任务...")
            sampling_tasks = prepare_sampling_tasks(dataset_name, target_tokens, dataset_info)
            
            if not sampling_tasks:
                logger.warning(f"  没有需要采样的文件，跳过")
                continue
            
            # 2.3 Shuffle任务以均衡负载
            random.seed(RANDOM_SEED)
            random.shuffle(sampling_tasks)
            logger.info(f"  已对任务列表进行随机打乱")
            
            # 2.4 分配任务到各个节点
            logger.info("  分配任务到各个节点...")
            node_task_lists = split_tasks_to_nodes(sampling_tasks, all_nodes)
            
            # 2.5 提交任务到所有节点
            logger.info("  提交任务到所有节点...")
            ray_tasks = submit_tasks_to_nodes(
                all_nodes=all_nodes,
                node_task_lists=node_task_lists,
                num_workers_per_node=NUM_WORKERS_PER_NODE
            )
            
            # 2.6 等待所有任务完成
            logger.info(f"  等待 {len(ray_tasks)} 个节点任务完成...")
            import time
            start_time = time.time()
            results = ray.get(ray_tasks)
            elapsed_time = time.time() - start_time
            
            # 2.7 汇总结果
            aggregate_results(results, dataset_name, target_tokens, output_path)
            
            logger.info(f"  数据集 {dataset_name} 处理耗时: {elapsed_time:.2f} 秒")
            
            # 释放内存
            del sampling_tasks
            del node_task_lists
            del results
            gc.collect()
        
        logger.info("\n" + "=" * 80)
        logger.info("所有数据集采样完成!")
        logger.info(f"输出目录: {output_path}")
        logger.info(f"采样使用的随机种子: {RANDOM_SEED}")
        logger.info("=" * 80)
        
    finally:
        # 关闭 Ray
        ray.shutdown()
        logger.info("Ray 已关闭")


if __name__ == "__main__":
    main()

