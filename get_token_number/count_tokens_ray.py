"""
使用 Ray 分布式框架计算数据集的 token 数量
每个节点使用多进程并行处理分配给它的文件
"""
import gc
import ray
import json
import random
from pathlib import Path
import time
import socket
import logging
from multiprocessing import Pool
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

# 导入工作函数模块
from get_token_number.count_single_parquet import count_tokens_in_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
DATA_ROOT = "/wl_intelligent/wangjinghui05/nvidia"
TOKENIZER_PATH = "/path/to/your/tokenizer"  # 请修改为实际的 tokenizer 路径
OUTPUT_JSON = "/wl_intelligent/shenzhiwei/token_statistics.json"
TEXT_COLUMN = "text"
BATCH_SIZE = 1000  # 批处理大小
NUM_WORKERS_PER_NODE = 10  # 每个节点的并行进程数
EXCLUDE_IPS = []  # 需要排除的节点IP列表
RANDOM_SEED = 42  # 随机种子，用于文件列表 shuffle


def find_all_datasets_and_files(data_root: str) -> Dict[str, Dict[str, List[str]]]:
    """
    扫描数据根目录，组织数据集结构
    
    Returns:
        {
            "dataset_name": {
                "subset_name": [file1.parquet, file2.parquet, ...],
                ...
            },
            ...
        }
    """
    data_root_path = Path(data_root)
    datasets = defaultdict(lambda: defaultdict(list))
    
    logger.info(f"扫描目录: {data_root}")
    
    # 遍历所有 parquet 文件
    parquet_files = list(data_root_path.rglob("*.parquet"))
    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    for parquet_file in tqdm(parquet_files, desc="组织文件结构"):
        # 获取相对路径
        rel_path = parquet_file.relative_to(data_root_path)
        parts = rel_path.parts

        # 处理不同层级结构：
        # - len(parts) == 1: 文件直接在根目录 -> dataset='root', subset='root'
        # - len(parts) == 2: dataset/file.parquet -> dataset=parts[0], subset='root'
        # - len(parts) == 3: dataset/subset/file.parquet -> dataset=parts[0], subset=parts[1]
        # - len(parts) >= 4: dataset/middle/subset/file.parquet -> dataset=parts[0], subset=parts[-2] (倒数第二个)
        if len(parts) == 1:
            # 文件直接在根目录
            dataset_name = "root"
            subset_name = "root"
        elif len(parts) == 2:
            # dataset/file.parquet
            dataset_name = parts[0]
            subset_name = "root"
        elif len(parts) == 3:
            # dataset/subset/file.parquet
            dataset_name = parts[0]
            subset_name = parts[1]
        else:
            # dataset/.../subset/file.parquet (多层，取第一层作为dataset，倒数第二层作为subset)
            dataset_name = parts[0]
            subset_name = parts[-2]  # 倒数第二个是子集目录名，最后一个是文件名

        datasets[dataset_name][subset_name].append(str(parquet_file))
    
    # 打印统计信息
    total_files = 0
    logger.info("\n数据集结构:")
    for dataset_name in sorted(datasets.keys()):
        subsets = datasets[dataset_name]
        dataset_files = sum(len(files) for files in subsets.values())
        total_files += dataset_files
        logger.info(f"  {dataset_name}:")
        for subset_name in sorted(subsets.keys()):
            logger.info(f"    - {subset_name}: {len(subsets[subset_name])} 文件")
    
    logger.info(f"\n总计: {len(datasets)} 个数据集, {total_files} 个文件")
    
    return dict(datasets)


def count_tokens_on_node(file_list: List[str], tokenizer_path: str, 
                         text_column: str, batch_size: int, num_workers: int) -> Dict:
    """
    在单个节点上使用多进程计算 token 数量
    
    Args:
        file_list: 需要处理的文件列表
        tokenizer_path: tokenizer 路径
        text_column: 文本列名
        batch_size: 批处理大小
        num_workers: 并行进程数
    
    Returns:
        处理结果统计信息
    """
    hostname = socket.gethostname()
    node_logger = logging.getLogger(f"Node-{hostname}")
    
    node_logger.info(f"节点 {hostname} 开始处理 {len(file_list)} 个文件")
    node_logger.info(f"使用 {num_workers} 个并行进程")
    
    # 准备参数列表
    args_list = [
        (file_path, tokenizer_path, text_column, batch_size, i % num_workers)
        for i, file_path in enumerate(file_list)
    ]
    
    # 统计信息
    file_stats = {}  # {file_path: (tokens, rows, error)}
    total_tokens = 0
    total_rows = 0
    success_count = 0
    failed_files = []
    
    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(count_tokens_in_file, args_list),
            total=len(args_list),
            desc=f"[{hostname}] 处理文件"
        ))
    
    # 汇总结果
    for file_path, token_count, row_count, error in results:
        file_stats[file_path] = (token_count, row_count, error)
        
        if error is None:
            total_tokens += token_count
            total_rows += row_count
            success_count += 1
        else:
            failed_files.append((file_path, error))
    
    # 清理进程池内存
    gc.collect()
    
    # 打印总结
    node_logger.info("=" * 80)
    node_logger.info(f"节点 {hostname} 处理完成!")
    node_logger.info(f"成功处理: {success_count}/{len(file_list)} 个文件")
    node_logger.info(f"总 token 数: {total_tokens:,}")
    node_logger.info(f"总行数: {total_rows:,}")
    
    if failed_files:
        node_logger.warning(f"失败的文件数: {len(failed_files)}")
        for file_path, error in failed_files[:5]:  # 只显示前5个
            node_logger.warning(f"  - {Path(file_path).name}: {error}")
    
    return {
        'hostname': hostname,
        'total_files': len(file_list),
        'success_count': success_count,
        'failed_count': len(failed_files),
        'failed_files': [fp for fp, _ in failed_files],
        'total_tokens': total_tokens,
        'total_rows': total_rows,
        'file_stats': file_stats
    }


@ray.remote
def count_tokens_node_task(file_list: List[str], tokenizer_path: str, 
                           text_column: str, batch_size: int, num_workers: int) -> Dict:
    """
    Ray 远程任务：在节点上计算 token 数量
    
    Args:
        file_list: 分配给该节点的文件列表
        tokenizer_path: tokenizer 路径
        text_column: 文本列名
        batch_size: 批处理大小
        num_workers: 每个节点的并行进程数
    
    Returns:
        处理结果
    """
    node_ip = ray.util.get_node_ip_address()
    hostname = socket.gethostname()
    
    try:
        result = count_tokens_on_node(
            file_list=file_list,
            tokenizer_path=tokenizer_path,
            text_column=text_column,
            batch_size=batch_size,
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
            'total_files': len(file_list),
            'success_count': 0,
            'failed_count': len(file_list),
            'total_tokens': 0,
            'total_rows': 0
        }


def get_all_nodes(exclude_ips: List[str]) -> List:
    """获取所有可用的 Ray 节点"""
    nodes = [
        node for node in ray.nodes() 
        if node['Alive'] and node.get('NodeManagerAddress', 'unknown') not in exclude_ips
    ]
    logger.info(f"发现 {len(nodes)} 个可用节点")
    return nodes


def split_files_to_nodes(all_files: List[str], all_nodes: List) -> List[List[str]]:
    """
    将文件列表均匀分配到所有节点
    
    Args:
        all_files: 所有需要处理的文件列表
        all_nodes: 所有可用的节点列表
    
    Returns:
        每个节点分配的文件列表
    """
    node_num = len(all_nodes)
    total_files = len(all_files)
    
    # 均匀划分
    base_size = total_files // node_num
    remainder = total_files % node_num
    
    node_file_lists = []
    start = 0
    
    for node_idx in range(node_num):
        current_size = base_size + (1 if node_idx < remainder else 0)
        end = start + current_size
        
        node_files = all_files[start:end]
        node_file_lists.append(node_files)
        
        logger.info(f"节点 {node_idx}: 分配 {len(node_files)} 个文件")
        start = end
    
    return node_file_lists


def submit_tasks_to_nodes(all_nodes: List, node_file_lists: List[List[str]], 
                         tokenizer_path: str, text_column: str, 
                         batch_size: int, num_workers_per_node: int) -> List:
    """
    向所有节点提交处理任务
    
    Returns:
        Ray 任务引用列表
    """
    ray_tasks = []
    
    for node_idx, (node, file_list) in enumerate(zip(all_nodes, node_file_lists)):
        if not file_list:
            logger.warning(f"节点 {node_idx} 没有分配文件，跳过")
            continue
        
        # 创建指定节点上的任务
        task = count_tokens_node_task.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], 
                soft=False
            )
        ).remote(
            file_list=file_list,
            tokenizer_path=tokenizer_path,
            text_column=text_column,
            batch_size=batch_size,
            num_workers=num_workers_per_node
        )
        
        ray_tasks.append(task)
        logger.info(f"任务已提交到节点 {node_idx} (NodeID: {node['NodeID']})")
    
    return ray_tasks


def organize_results_by_dataset(datasets: Dict, all_results: List[Dict]) -> Dict:
    """
    按数据集和子集组织结果
    
    Args:
        datasets: 数据集结构 {dataset: {subset: [files]}}
        all_results: 所有节点返回的结果列表
    
    Returns:
        组织后的结果字典
    """
    # 合并所有节点的文件统计
    all_file_stats = {}
    for result in all_results:
        if result['status'] == 'success':
            all_file_stats.update(result.get('file_stats', {}))
    
    # 按数据集和子集组织
    organized_results = {
        'total_tokens': 0,
        'total_rows': 0,
        'total_files': 0,
        'failed_files': 0,
        'datasets': {}
    }
    
    for dataset_name, subsets in datasets.items():
        dataset_stats = {
            'total_tokens': 0,
            'total_rows': 0,
            'total_files': 0,
            'failed_files': 0,
            'subsets': {}
        }
        
        for subset_name, files in subsets.items():
            subset_tokens = 0
            subset_rows = 0
            subset_failed = 0
            
            for file_path in files:
                if file_path in all_file_stats:
                    tokens, rows, error = all_file_stats[file_path]
                    if error is None:
                        subset_tokens += tokens
                        subset_rows += rows
                    else:
                        subset_failed += 1
            
            dataset_stats['subsets'][subset_name] = {
                'token_count': subset_tokens,
                'row_count': subset_rows,
                'file_count': len(files),
                'failed_files': subset_failed
            }
            
            dataset_stats['total_tokens'] += subset_tokens
            dataset_stats['total_rows'] += subset_rows
            dataset_stats['total_files'] += len(files)
            dataset_stats['failed_files'] += subset_failed
        
        organized_results['datasets'][dataset_name] = dataset_stats
        organized_results['total_tokens'] += dataset_stats['total_tokens']
        organized_results['total_rows'] += dataset_stats['total_rows']
        organized_results['total_files'] += dataset_stats['total_files']
        organized_results['failed_files'] += dataset_stats['failed_files']
    
    return organized_results


def print_summary(results: Dict):
    """打印汇总信息"""
    logger.info("\n" + "=" * 80)
    logger.info("Token 统计汇总")
    logger.info("=" * 80)
    logger.info(f"总 token 数: {results['total_tokens']:,}")
    logger.info(f"总行数: {results['total_rows']:,}")
    logger.info(f"总文件数: {results['total_files']:,}")
    logger.info(f"失败文件数: {results['failed_files']}")
    logger.info("=" * 80)
    
    logger.info("\n各数据集统计:")
    for dataset_name, dataset_stats in sorted(results['datasets'].items()):
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  总 token 数: {dataset_stats['total_tokens']:,}")
        logger.info(f"  总行数: {dataset_stats['total_rows']:,}")
        logger.info(f"  文件数: {dataset_stats['total_files']}")
        
        # 计算占比
        if results['total_tokens'] > 0:
            percentage = (dataset_stats['total_tokens'] / results['total_tokens']) * 100
            logger.info(f"  占比: {percentage:.2f}%")
        
        # 打印子集信息
        logger.info(f"  子集 ({len(dataset_stats['subsets'])}):")
        for subset_name, subset_stats in sorted(dataset_stats['subsets'].items(), 
                                                 key=lambda x: x[1]['token_count'], 
                                                 reverse=True):
            logger.info(f"    - {subset_name}: {subset_stats['token_count']:,} tokens, "
                       f"{subset_stats['row_count']:,} rows, "
                       f"{subset_stats['file_count']} files")


def main():
    """主函数"""
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("Ray 分布式 Token 计数工具")
    logger.info("=" * 80)
    logger.info(f"数据根目录: {DATA_ROOT}")
    logger.info(f"Tokenizer 路径: {TOKENIZER_PATH}")
    logger.info(f"输出文件: {OUTPUT_JSON}")
    logger.info(f"文本列名: {TEXT_COLUMN}")
    logger.info(f"批处理大小: {BATCH_SIZE}")
    logger.info(f"每节点进程数: {NUM_WORKERS_PER_NODE}")
    logger.info("=" * 80)
    
    # 检查数据目录
    if not Path(DATA_ROOT).exists():
        logger.error(f"数据目录不存在: {DATA_ROOT}")
        return
    
    # 初始化 Ray
    logger.info("\n连接到 Ray 集群...")
    ray.init(address="auto")
    
    cluster_resources = ray.cluster_resources()
    num_cpus = int(cluster_resources.get('CPU', 0))
    num_nodes = len(ray.nodes())
    
    logger.info(f"Ray 集群信息:")
    logger.info(f"  节点数: {num_nodes}")
    logger.info(f"  总 CPU 数: {num_cpus}")
    
    try:
        # 扫描所有数据集和文件
        logger.info("\n步骤 1: 扫描数据集...")
        datasets = find_all_datasets_and_files(DATA_ROOT)
        
        # 获取所有文件的扁平列表
        all_files = []
        for subsets in datasets.values():
            for files in subsets.values():
                all_files.extend(files)
        
        # Shuffle 文件列表以均衡负载
        random.seed(RANDOM_SEED)
        random.shuffle(all_files)
        logger.info(f"已对文件列表进行随机打乱 (seed={RANDOM_SEED})")
        
        logger.info(f"\n总计需要处理 {len(all_files)} 个文件")
        
        # 获取所有可用节点
        all_nodes = get_all_nodes(EXCLUDE_IPS)
        if not all_nodes:
            logger.error("没有可用的节点!")
            return
        
        # 分配文件到各个节点
        logger.info("\n步骤 2: 分配文件到各个节点...")
        node_file_lists = split_files_to_nodes(all_files, all_nodes)
        
        # 提交任务到所有节点
        logger.info("\n步骤 3: 提交任务到所有节点...")
        ray_tasks = submit_tasks_to_nodes(
            all_nodes=all_nodes,
            node_file_lists=node_file_lists,
            tokenizer_path=TOKENIZER_PATH,
            text_column=TEXT_COLUMN,
            batch_size=BATCH_SIZE,
            num_workers_per_node=NUM_WORKERS_PER_NODE
        )
        
        # 等待所有任务完成
        logger.info(f"\n步骤 4: 等待 {len(ray_tasks)} 个节点任务完成...")
        task_start_time = time.time()
        all_results = ray.get(ray_tasks)
        task_elapsed_time = time.time() - task_start_time
        
        logger.info(f"所有节点任务完成! 耗时: {task_elapsed_time:.2f} 秒")
        
        # 组织结果
        logger.info("\n步骤 5: 组织和汇总结果...")
        organized_results = organize_results_by_dataset(datasets, all_results)
        
        # 清理大对象内存
        del all_results
        gc.collect()
        
        # 添加元数据
        organized_results['metadata'] = {
            'data_root': DATA_ROOT,
            'tokenizer_path': TOKENIZER_PATH,
            'text_column': TEXT_COLUMN,
            'batch_size': BATCH_SIZE,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time_seconds': time.time() - start_time,
            'num_nodes': num_nodes,
            'num_workers_per_node': NUM_WORKERS_PER_NODE
        }
        
        # 保存结果
        logger.info("\n步骤 6: 保存结果...")
        output_path = Path(OUTPUT_JSON)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(organized_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {output_path}")
        
        # 打印汇总
        print_summary(organized_results)
        
        total_elapsed_time = time.time() - start_time
        logger.info(f"\n总耗时: {total_elapsed_time:.2f} 秒")
        logger.info(f"平均速度: {organized_results['total_tokens']/(total_elapsed_time/60):.0f} tokens/分钟")
        logger.info("\n所有任务完成!")
        
    finally:
        # 关闭 Ray
        ray.shutdown()
        logger.info("Ray 已关闭")


if __name__ == "__main__":
    main()
