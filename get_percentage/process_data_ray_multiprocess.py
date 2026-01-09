"""
使用 Ray 分布式框架在多个节点上并行处理数据
每个节点内部使用多进程处理分配给它的数据文件
支持格式: .parquet, .jsonl.zst
"""
import ray
import json
from pathlib import Path
import time
import socket
import logging
import shutil
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# 导入工作函数模块
from get_percentage.sample_single_parquet import process_single_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
SOURCE_DIR = "/wl_intelligent/wangjinghui05/nvidia"
TARGET_DIR = "/wl_intelligent/shenzhiwei/model_data/nvidia_distributed"
SAMPLE_RATIO = 0.05  # 5%
RANDOM_SEED = 42
NUM_WORKERS_PER_NODE = 10  # 每个节点的并行进程数
EXCLUDE_IPS = ['10.48.90.208']  # 需要排除的节点IP列表


def find_all_parquet_files(source_dir, random_seed=42):
    """递归查找所有 parquet 文件、.jsonl.zst 文件和 README.md 文件"""
    import random
    
    parquet_files = []
    readme_files = []
    source_path = Path(source_dir)
    
    logger.info(f"正在扫描目录: {source_dir}")
    
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            if file_path.suffix == ".parquet":
                parquet_files.append(str(file_path))
            elif file_path.suffix == ".zst" and file_path.name.endswith('.jsonl.zst'):
                parquet_files.append(str(file_path))
            elif file_path.name == "README.md":
                readme_files.append(str(file_path))
    
    logger.info(f"找到 {len(parquet_files)} 个数据文件 (parquet + jsonl.zst)")
    logger.info(f"找到 {len(readme_files)} 个 README.md 文件")
    
    # Shuffle 文件列表（使用随机种子保证可重复性）
    random.seed(random_seed)
    random.shuffle(parquet_files)
    logger.info(f"已对文件列表进行随机打乱 (seed={random_seed})")
    
    return parquet_files, readme_files


def copy_readme_files(readme_files, source_dir, target_dir):
    """
    复制所有 README.md 文件到对应的目标目录位置
    
    Args:
        readme_files: README.md 文件列表
        source_dir: 源目录
        target_dir: 目标目录
    """
    logger.info(f"\n开始复制 {len(readme_files)} 个 README.md 文件...")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    success_count = 0
    failed_count = 0
    
    for readme_file in tqdm(readme_files, desc="复制 README.md"):
        try:
            # 计算相对路径
            readme_path = Path(readme_file)
            relative_path = readme_path.relative_to(source_path)
            
            # 计算目标路径
            target_file = target_path / relative_path
            
            # 创建目标目录
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(readme_file, target_file)
            success_count += 1
            
        except Exception as e:
            logger.error(f"复制 README.md 失败: {readme_file}, 错误: {str(e)}")
            failed_count += 1
    
    logger.info(f"README.md 文件复制完成: 成功 {success_count}, 失败 {failed_count}")



def process_files_on_node(file_list, source_dir, target_dir, sample_ratio, random_seed, num_workers):
    """
    在单个节点上使用多进程处理文件列表
    
    Args:
        file_list: 需要处理的文件列表
        source_dir: 源目录
        target_dir: 目标目录
        sample_ratio: 抽样比例
        random_seed: 随机种子
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
        (source_file, source_dir, target_dir, sample_ratio, random_seed, i % num_workers)
        for i, source_file in enumerate(file_list)
    ]
    
    # 统计信息
    total_original = 0
    total_sampled = 0
    success_count = 0
    failed_files = []
    
    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, args_list),
            total=len(args_list),
            desc=f"[{hostname}] 处理文件"
        ))
    
    # 汇总结果
    for original_count, sampled_count, failed_file in results:
        total_original += original_count
        total_sampled += sampled_count
        if failed_file is None:
            if original_count > 0 or sampled_count > 0:
                success_count += 1
        else:
            failed_files.append(failed_file)
    
    # 打印总结
    node_logger.info("=" * 80)
    node_logger.info(f"节点 {hostname} 处理完成!")
    node_logger.info(f"成功处理: {success_count}/{len(file_list)} 个文件")
    node_logger.info(f"总原始数据量: {total_original:,}")
    node_logger.info(f"总抽样数据量: {total_sampled:,}")
    if total_original > 0:
        node_logger.info(f"实际抽样比例: {total_sampled/total_original*100:.2f}%")
    
    if failed_files:
        node_logger.warning(f"失败的文件数: {len(failed_files)}")
    
    return {
        'hostname': hostname,
        'total_files': len(file_list),
        'success_count': success_count,
        'failed_count': len(failed_files),
        'failed_files': failed_files,
        'total_original': total_original,
        'total_sampled': total_sampled
    }


@ray.remote
def process_node_task(file_list, source_dir, target_dir, sample_ratio, random_seed, num_workers):
    """
    Ray远程任务：在节点上处理数据
    
    Args:
        file_list: 分配给该节点的文件列表
        source_dir: 源目录
        target_dir: 目标目录
        sample_ratio: 抽样比例
        random_seed: 随机种子
        num_workers: 每个节点的并行进程数
    
    Returns:
        处理结果
    """
    node_ip = ray.util.get_node_ip_address()
    hostname = socket.gethostname()
    
    try:
        result = process_files_on_node(
            file_list=file_list,
            source_dir=source_dir,
            target_dir=target_dir,
            sample_ratio=sample_ratio,
            random_seed=random_seed,
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
            'error': str(e)
        }


def get_all_nodes(exclude_ips):
    """获取所有可用的Ray节点"""
    nodes = [
        node for node in ray.nodes() 
        if node['Alive'] and node.get('NodeManagerAddress', 'unknown') not in exclude_ips
    ]
    logger.info(f"发现 {len(nodes)} 个可用节点")
    return nodes


def split_files_to_nodes(all_files, all_nodes):
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
    
    # 均匀划分：前 remainder 个节点多分配 1 个文件
    base_size = total_files // node_num
    remainder = total_files % node_num
    
    node_file_lists = []
    start = 0
    
    for node_idx in range(node_num):
        # 前 remainder 个节点多分配 1 个元素
        current_size = base_size + (1 if node_idx < remainder else 0)
        end = start + current_size
        
        node_files = all_files[start:end]
        node_file_lists.append(node_files)
        
        logger.info(f"节点 {node_idx}: 分配 {len(node_files)} 个文件, 范围 [{start}, {end-1}]")
        start = end
    
    return node_file_lists


def submit_tasks_to_nodes(all_nodes, node_file_lists, source_dir, target_dir, 
                          sample_ratio, random_seed, num_workers_per_node):
    """
    向所有节点提交处理任务
    
    Args:
        all_nodes: 所有节点列表
        node_file_lists: 每个节点分配的文件列表
        source_dir: 源目录
        target_dir: 目标目录
        sample_ratio: 抽样比例
        random_seed: 随机种子
        num_workers_per_node: 每个节点的并行进程数
    
    Returns:
        Ray任务引用列表
    """
    ray_tasks = []
    
    for node_idx, (node, file_list) in enumerate(zip(all_nodes, node_file_lists)):
        if not file_list:
            logger.warning(f"节点 {node_idx} 没有分配文件，跳过")
            continue
        
        # 创建指定节点上的任务
        task = process_node_task.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], 
                soft=False
            )
        ).remote(
            file_list=file_list,
            source_dir=source_dir,
            target_dir=target_dir,
            sample_ratio=sample_ratio,
            random_seed=random_seed,
            num_workers=num_workers_per_node
        )
        
        ray_tasks.append(task)
        logger.info(f"任务已提交到节点 {node_idx} (NodeID: {node['NodeID']})")
    
    return ray_tasks


def aggregate_results(results):
    """
    汇总所有节点的处理结果
    
    Args:
        results: 所有节点返回的结果列表
    """
    logger.info("=" * 80)
    logger.info("所有节点处理完成，汇总结果:")
    logger.info("=" * 80)
    
    total_files = 0
    total_success = 0
    total_failed = 0
    total_original = 0
    total_sampled = 0
    all_failed_files = []
    
    for idx, result in enumerate(results):
        if result['status'] == 'failed':
            logger.error(f"节点 {result['hostname']} ({result['node_ip']}) 处理失败: {result['error']}")
            continue
        
        logger.info(f"\n节点 {idx + 1}: {result['hostname']} ({result['node_ip']})")
        logger.info(f"  处理文件数: {result['total_files']}")
        logger.info(f"  成功: {result['success_count']}, 失败: {result['failed_count']}")
        logger.info(f"  原始数据量: {result['total_original']:,}")
        logger.info(f"  抽样数据量: {result['total_sampled']:,}")
        
        total_files += result['total_files']
        total_success += result['success_count']
        total_failed += result['failed_count']
        total_original += result['total_original']
        total_sampled += result['total_sampled']
        all_failed_files.extend(result['failed_files'])
    
    logger.info("\n" + "=" * 80)
    logger.info("总体统计:")
    logger.info(f"  总文件数: {total_files}")
    logger.info(f"  成功处理: {total_success}")
    logger.info(f"  处理失败: {total_failed}")
    logger.info(f"  总原始数据量: {total_original:,}")
    logger.info(f"  总抽样数据量: {total_sampled:,}")
    if total_original > 0:
        logger.info(f"  实际抽样比例: {total_sampled/total_original*100:.2f}%")
    logger.info("=" * 80)
    
    if all_failed_files:
        logger.warning(f"\n失败的文件总数: {len(all_failed_files)}")
        logger.warning("失败文件列表:")
        for failed_file in all_failed_files[:20]:  # 只显示前20个
            logger.warning(f"  - {failed_file}")
        if len(all_failed_files) > 20:
            logger.warning(f"  ... 还有 {len(all_failed_files) - 20} 个失败文件")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Ray 分布式数据处理工具")
    logger.info("=" * 80)
    logger.info(f"源目录: {SOURCE_DIR}")
    logger.info(f"目标目录: {TARGET_DIR}")
    logger.info(f"抽样比例: {SAMPLE_RATIO * 100}%")
    logger.info(f"随机种子: {RANDOM_SEED}")
    logger.info(f"每节点进程数: {NUM_WORKERS_PER_NODE}")
    logger.info("=" * 80)
    
    # 检查源目录
    if not Path(SOURCE_DIR).exists():
        logger.error(f"源目录不存在: {SOURCE_DIR}")
        return
    
    # 创建目标目录
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"目标目录已创建: {TARGET_DIR}")
    
    # 初始化 Ray（指定运行时环境，自动安装依赖）
    logger.info("\n连接到 Ray 集群...")
    ray.init(
        address="auto",
        runtime_env={
            "pip": ["zstandard"]  # 自动在所有节点安装 zstandard
        }
    )
    logger.info("Ray 连接成功!")
    
    try:
        # 获取所有可用节点
        all_nodes = get_all_nodes(EXCLUDE_IPS)
        if not all_nodes:
            logger.error("没有可用的节点!")
            return
        
        # 查找所有需要处理的文件
        logger.info("\n开始扫描文件...")
        all_files, readme_files = find_all_parquet_files(SOURCE_DIR, RANDOM_SEED)
        if not all_files:
            logger.warning("没有找到任何数据文件 (parquet 或 jsonl.zst)")
            return
        
        # 复制所有 README.md 文件
        if readme_files:
            copy_readme_files(readme_files, SOURCE_DIR, TARGET_DIR)
        else:
            logger.info("没有找到 README.md 文件")
        
        # 分配文件到各个节点
        logger.info("\n分配文件到各个节点...")
        node_file_lists = split_files_to_nodes(all_files, all_nodes)
        
        # 提交任务到所有节点
        logger.info("\n提交任务到所有节点...")
        ray_tasks = submit_tasks_to_nodes(
            all_nodes=all_nodes,
            node_file_lists=node_file_lists,
            source_dir=SOURCE_DIR,
            target_dir=TARGET_DIR,
            sample_ratio=SAMPLE_RATIO,
            random_seed=RANDOM_SEED,
            num_workers_per_node=NUM_WORKERS_PER_NODE
        )
        
        # 等待所有任务完成
        logger.info(f"\n等待 {len(ray_tasks)} 个节点任务完成...")
        start_time = time.time()
        results = ray.get(ray_tasks)
        elapsed_time = time.time() - start_time
        
        # 汇总结果
        aggregate_results(results)
        
        logger.info(f"\n总耗时: {elapsed_time:.2f} 秒")
        logger.info("所有任务完成!")
        
    finally:
        # 关闭 Ray
        ray.shutdown()
        logger.info("Ray 已关闭")


if __name__ == "__main__":
    main()
