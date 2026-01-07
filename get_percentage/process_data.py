"""
从 Nvidia 数据集中抽取 5% 的数据
保持原有目录结构,每个 parquet 文件 shuffle 后抽取 5%
"""
import os
import gc
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import logging
from multiprocessing import Pool, Manager
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
SOURCE_DIR = "/wl_intelligent/wangjinghui05/nvidia"
TARGET_DIR = "/wl_intelligent/shenzhiwei/model_data/nvidia"
SAMPLE_RATIO = 0.05  # 5%
RANDOM_SEED = 42
NUM_WORKERS = 4  # 并行进程数


def find_all_parquet_files(source_dir):
    """递归查找所有 parquet 文件"""
    parquet_files = []
    source_path = Path(source_dir)
    
    logger.info(f"正在扫描目录: {source_dir}")
    
    for file_path in source_path.rglob("*.parquet"):
        parquet_files.append(file_path)
    
    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
    return parquet_files


def get_relative_path(file_path, source_dir):
    """获取相对于源目录的相对路径"""
    return Path(file_path).relative_to(source_dir)


def create_target_path(relative_path, target_dir):
    """创建目标文件路径"""
    target_path = Path(target_dir) / relative_path
    # 创建目标目录
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def sample_parquet_file(source_file, target_file, sample_ratio=0.05, random_seed=42, worker_id=None):
    """
    从单个 parquet 文件中抽取样本
    
    Args:
        source_file: 源文件路径
        target_file: 目标文件路径
        sample_ratio: 抽样比例
        random_seed: 随机种子
        worker_id: 工作进程ID(用于日志)
    """
    # 为每个进程配置独立的 logger
    worker_logger = logging.getLogger(f"Worker-{worker_id}" if worker_id is not None else "Main")
    
    try:
        # 读取 parquet 文件
        worker_logger.info(f"正在读取: {source_file}")
        df = pd.read_parquet(source_file)
        
        original_count = len(df)
        worker_logger.info(f"原始数据量: {original_count}")
        
        if original_count == 0:
            worker_logger.warning(f"文件为空: {source_file}")
            # 创建空的 parquet 文件
            df.to_parquet(target_file, index=False)
            del df
            gc.collect()
            return 0, 0, None
        
        # Shuffle 并抽取样本
        df_sampled = df.sample(
            frac=sample_ratio,
            random_state=random_seed,
            replace=False
        ).reset_index(drop=True)
        
        sampled_count = len(df_sampled)
        worker_logger.info(f"抽样数据量: {sampled_count} ({sampled_count/original_count*100:.2f}%)")
        
        # 保存到目标文件
        df_sampled.to_parquet(target_file, index=False)
        worker_logger.info(f"已保存到: {target_file}")
        
        # 立即释放内存
        del df
        del df_sampled
        gc.collect()
        
        return original_count, sampled_count, None
        
    except Exception as e:
        worker_logger.error(f"处理文件失败 {source_file}: {str(e)}")
        return 0, 0, str(source_file)


def process_single_file(args):
    """
    处理单个文件的包装函数(用于多进程)
    
    Args:
        args: (source_file, source_dir, target_dir, sample_ratio, random_seed, worker_id)
    
    Returns:
        (original_count, sampled_count, failed_file)
    """
    source_file, source_dir, target_dir, sample_ratio, random_seed, worker_id = args
    
    try:
        # 获取相对路径
        relative_path = get_relative_path(source_file, source_dir)
        
        # 创建目标路径
        target_file = create_target_path(relative_path, target_dir)
        
        # 检查目标文件是否已存在
        if target_file.exists():
            worker_logger = logging.getLogger(f"Worker-{worker_id}")
            worker_logger.info(f"目标文件已存在，跳过: {target_file}")
            return 0, 0, None
        
        # 处理文件
        original_count, sampled_count, failed_file = sample_parquet_file(
            source_file, 
            target_file, 
            sample_ratio, 
            random_seed,
            worker_id
        )
        
        return original_count, sampled_count, failed_file
        
    except Exception as e:
        worker_logger = logging.getLogger(f"Worker-{worker_id}")
        worker_logger.error(f"处理失败: {source_file}")
        worker_logger.error(f"错误信息: {str(e)}")
        return 0, 0, str(source_file)


def process_all_files(source_dir, target_dir, sample_ratio=0.05, random_seed=42, num_workers=4):
    """
    并行处理所有 parquet 文件
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        sample_ratio: 抽样比例
        random_seed: 随机种子
        num_workers: 并行进程数
    """
    # 查找所有 parquet 文件
    parquet_files = find_all_parquet_files(source_dir)
    
    if not parquet_files:
        logger.warning("没有找到任何 parquet 文件")
        return
    
    logger.info(f"使用 {num_workers} 个并行进程处理文件")
    
    # 准备参数列表
    args_list = [
        (source_file, source_dir, target_dir, sample_ratio, random_seed, i % num_workers)
        for i, source_file in enumerate(parquet_files)
    ]
    
    # 统计信息
    total_original = 0
    total_sampled = 0
    success_count = 0
    failed_files = []
    
    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 以获得更好的性能和进度显示
        results = list(tqdm(
            pool.imap_unordered(process_single_file, args_list),
            total=len(args_list),
            desc="处理文件"
        ))
    
    # 汇总结果
    for original_count, sampled_count, failed_file in results:
        total_original += original_count
        total_sampled += sampled_count
        if failed_file is None:
            if original_count > 0 or sampled_count > 0:  # 排除跳过的文件
                success_count += 1
        else:
            failed_files.append(failed_file)
    
    # 打印总结
    logger.info("=" * 80)
    logger.info("处理完成!")
    logger.info(f"成功处理: {success_count}/{len(parquet_files)} 个文件")
    logger.info(f"总原始数据量: {total_original:,}")
    logger.info(f"总抽样数据量: {total_sampled:,}")
    if total_original > 0:
        logger.info(f"实际抽样比例: {total_sampled/total_original*100:.2f}%")
    
    if failed_files:
        logger.warning(f"失败的文件数: {len(failed_files)}")
        logger.warning("失败的文件列表:")
        for failed_file in failed_files:
            logger.warning(f"  - {failed_file}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Nvidia 数据集抽样工具 (并行版本)")
    logger.info("=" * 80)
    logger.info(f"源目录: {SOURCE_DIR}")
    logger.info(f"目标目录: {TARGET_DIR}")
    logger.info(f"抽样比例: {SAMPLE_RATIO * 100}%")
    logger.info(f"随机种子: {RANDOM_SEED}")
    logger.info(f"并行进程数: {NUM_WORKERS}")
    logger.info("=" * 80)
    
    # 检查源目录是否存在
    if not Path(SOURCE_DIR).exists():
        logger.error(f"源目录不存在: {SOURCE_DIR}")
        return
    
    # 创建目标目录
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"目标目录已创建: {TARGET_DIR}")
    
    # 处理所有文件
    process_all_files(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        sample_ratio=SAMPLE_RATIO,
        random_seed=RANDOM_SEED,
        num_workers=NUM_WORKERS
    )
    
    logger.info("所有任务完成!")


if __name__ == "__main__":
    main()
