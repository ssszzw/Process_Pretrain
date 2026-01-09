"""
工作函数模块：包含所有需要在多进程中执行的函数
这些函数定义在独立模块中，可以被 multiprocessing 正确序列化
"""
from pathlib import Path
import pandas as pd
import json
import gc
import logging
from utils.path_func import get_relative_path, create_target_path

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


def read_jsonl_zst(file_path):
    """
    读取 zstd 压缩的 jsonl 文件
    
    Args:
        file_path: .jsonl.zst 文件路径
    
    Returns:
        包含所有记录的列表
    """
    if not ZSTD_AVAILABLE:
        raise ImportError("zstandard 库未安装，请运行: pip install zstandard")
    
    data = []
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode('utf-8')
            for line in text_stream.strip().split('\n'):
                if line.strip():
                    data.append(json.loads(line))
    return data



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


def sample_jsonl_zst_file(source_file, target_file, sample_ratio=0.05, random_seed=42, worker_id=None):
    """
    从单个 .jsonl.zst 文件中抽取样本并保存为 parquet 格式
    
    Args:
        source_file: 源文件路径 (.jsonl.zst)
        target_file: 目标文件路径 (.parquet)
        sample_ratio: 抽样比例
        random_seed: 随机种子
        worker_id: 工作进程ID(用于日志)
    """
    worker_logger = logging.getLogger(f"Worker-{worker_id}" if worker_id is not None else "Main")
    
    try:
        # 读取 .jsonl.zst 文件
        worker_logger.info(f"正在读取: {source_file}")
        data = read_jsonl_zst(source_file)
        
        # 转换为 DataFrame
        df = pd.DataFrame(data)
        
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
        
        # 保存到目标文件（parquet 格式）
        df_sampled.to_parquet(target_file, index=False)
        worker_logger.info(f"已保存到: {target_file}")
        
        # 立即释放内存
        del df
        del df_sampled
        del data
        gc.collect()
        
        return original_count, sampled_count, None
        
    except Exception as e:
        worker_logger.error(f"处理文件失败 {source_file}: {str(e)}")
        import traceback
        worker_logger.error(traceback.format_exc())
        return 0, 0, str(source_file)


def process_single_file(args):
    """
    处理单个文件的包装函数(用于多进程)
    支持 .parquet 和 .jsonl.zst 格式
    
    Args:
        args: (source_file, source_dir, target_dir, sample_ratio, random_seed, worker_id)
    
    Returns:
        (original_count, sampled_count, failed_file)
    """
    source_file, source_dir, target_dir, sample_ratio, random_seed, worker_id = args
    
    try:
        # 获取相对路径
        relative_path = get_relative_path(source_file, source_dir)
        
        # 判断文件类型并确定目标路径
        source_path_str = str(relative_path)
        if source_path_str.endswith('.jsonl.zst'):
            # .jsonl.zst 转换为 .parquet
            target_path_str = source_path_str[:-11] + '.parquet'
            is_jsonl_zst = True
        elif source_path_str.endswith('.parquet'):
            # .parquet 保持不变
            target_path_str = source_path_str
            is_jsonl_zst = False
        else:
            # 其他格式，保持原样
            target_path_str = source_path_str
            is_jsonl_zst = False
        
        relative_path = Path(target_path_str)
        
        # 创建目标路径
        target_file = create_target_path(relative_path, target_dir)
        
        # 检查目标文件是否已存在
        if target_file.exists():
            return 0, 0, None
        
        # 根据文件类型选择处理函数
        if is_jsonl_zst:
            original_count, sampled_count, failed_file = sample_jsonl_zst_file(
                source_file, 
                target_file, 
                sample_ratio, 
                random_seed,
                worker_id
            )
        else:
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
        import traceback
        worker_logger.error(traceback.format_exc())
        return 0, 0, str(source_file)
