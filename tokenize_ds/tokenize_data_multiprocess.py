"""
单机版本的tokenize工具
使用多进程在单机上并行tokenize数据
支持格式: .parquet, .jsonl.zst
将 'text' 或 'content' 字段的内容tokenize后替换原字段
"""
import json
from pathlib import Path
import time
import logging
import shutil
from multiprocessing import Pool
from tqdm import tqdm

# 导入工作函数模块
from tokenize_ds.tokenize_single_file import process_single_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
SOURCE_DIR = "/path/to/source"  # 源目录，需要修改
TARGET_DIR = "/path/to/target"  # 目标目录，需要修改
TOKENIZER_PATH = "/path/to/tokenizer"  # tokenizer路径，需要修改
TEXT_COLUMNS = ["text", "content"]  # 文本字段名称列表，按优先级依次查找
NUM_WORKERS = 10  # 并行进程数


def find_all_data_files(source_dir):
    """递归查找所有 parquet 文件、.jsonl.zst 文件和 README.md 文件"""
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


def process_files(file_list, source_dir, target_dir, tokenizer_path, text_columns, num_workers):
    """
    使用多进程处理文件列表
    
    Args:
        file_list: 需要处理的文件列表
        source_dir: 源目录
        target_dir: 目标目录
        tokenizer_path: tokenizer路径
        text_columns: 文本字段名称列表，按优先级依次查找
        num_workers: 并行进程数
    
    Returns:
        处理结果统计信息
    """
    logger.info(f"开始处理 {len(file_list)} 个文件")
    logger.info(f"使用 {num_workers} 个并行进程")
    logger.info(f"文本字段查找顺序: {text_columns}")
    
    # 准备参数列表
    args_list = [
        (source_file, source_dir, target_dir, tokenizer_path, text_columns, i % num_workers)
        for i, source_file in enumerate(file_list)
    ]
    
    # 统计信息
    total_original = 0
    total_processed = 0
    success_count = 0
    failed_files = []
    
    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, args_list),
            total=len(args_list),
            desc="处理文件"
        ))
    
    # 汇总结果
    for original_count, processed_count, failed_file in results:
        total_original += original_count
        total_processed += processed_count
        if failed_file is None:
            if original_count > 0 or processed_count > 0:
                success_count += 1
        else:
            failed_files.append(failed_file)
    
    # 打印总结
    logger.info("=" * 80)
    logger.info("处理完成!")
    logger.info(f"成功处理: {success_count}/{len(file_list)} 个文件")
    logger.info(f"总原始数据量: {total_original:,}")
    logger.info(f"总处理数据量: {total_processed:,}")
    
    if failed_files:
        logger.warning(f"失败的文件数: {len(failed_files)}")
        logger.warning("失败文件列表:")
        for failed_file in failed_files[:20]:  # 只显示前20个
            logger.warning(f"  - {failed_file}")
        if len(failed_files) > 20:
            logger.warning(f"  ... 还有 {len(failed_files) - 20} 个失败文件")
    
    return {
        'total_files': len(file_list),
        'success_count': success_count,
        'failed_count': len(failed_files),
        'failed_files': failed_files,
        'total_original': total_original,
        'total_processed': total_processed
    }


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("数据Tokenize工具（单机版）")
    logger.info("=" * 80)
    logger.info(f"源目录: {SOURCE_DIR}")
    logger.info(f"目标目录: {TARGET_DIR}")
    logger.info(f"Tokenizer路径: {TOKENIZER_PATH}")
    logger.info(f"文本字段查找顺序: {TEXT_COLUMNS}")
    logger.info(f"并行进程数: {NUM_WORKERS}")
    logger.info("=" * 80)
    
    # 检查源目录
    if not Path(SOURCE_DIR).exists():
        logger.error(f"源目录不存在: {SOURCE_DIR}")
        return
    
    # 检查tokenizer路径
    if not Path(TOKENIZER_PATH).exists():
        logger.error(f"Tokenizer路径不存在: {TOKENIZER_PATH}")
        return
    
    # 创建目标目录
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"目标目录已创建: {TARGET_DIR}")
    
    # 查找所有需要处理的文件
    logger.info("\n开始扫描文件...")
    all_files, readme_files = find_all_data_files(SOURCE_DIR)
    if not all_files:
        logger.warning("没有找到任何数据文件 (parquet 或 jsonl.zst)")
        return
    
    # 复制所有 README.md 文件
    if readme_files:
        copy_readme_files(readme_files, SOURCE_DIR, TARGET_DIR)
    else:
        logger.info("没有找到 README.md 文件")
    
    # 处理所有文件
    logger.info("\n开始处理文件...")
    start_time = time.time()
    result = process_files(
        file_list=all_files,
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        tokenizer_path=TOKENIZER_PATH,
        text_columns=TEXT_COLUMNS,
        num_workers=NUM_WORKERS
    )
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("总体统计:")
    logger.info(f"  总文件数: {result['total_files']}")
    logger.info(f"  成功处理: {result['success_count']}")
    logger.info(f"  处理失败: {result['failed_count']}")
    logger.info(f"  总原始数据量: {result['total_original']:,}")
    logger.info(f"  总处理数据量: {result['total_processed']:,}")
    logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
    logger.info("=" * 80)
    
    logger.info("\n所有任务完成!")


if __name__ == "__main__":
    main()
