"""
采样单个 parquet 文件
用于多进程并行处理
"""
import gc
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

# 配置日志
logger = logging.getLogger(__name__)


def sample_single_file(args) -> Tuple[bool, int, int, str]:
    """
    采样单个文件的工作函数（用于多进程）
    
    Args:
        args: (parquet_path, target_tokens, output_path, file_seed, worker_id)
    
    Returns:
        (success, actual_tokens, row_count, output_path)
    """
    parquet_path, target_tokens, output_path, file_seed, worker_id = args
    
    # 为每个进程配置独立的 logger
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        
        if 'token_number' not in df.columns:
            worker_logger.warning(f"文件缺少 token_number 字段: {parquet_path}")
            return False, 0, 0, output_path
        
        # 随机打乱数据
        df = df.sample(frac=1, random_state=file_seed).reset_index(drop=True)
        
        # 累计采样直到达到目标token量
        cumsum_tokens = df['token_number'].cumsum()
        sampled_indices = cumsum_tokens <= target_tokens
        
        # 如果没有任何数据符合条件，至少取第一条
        if not sampled_indices.any():
            sampled_df = df.iloc[:1]
        else:
            last_idx = sampled_indices.sum()
            sampled_df = df.iloc[:last_idx]
        
        actual_tokens = int(sampled_df['token_number'].sum())
        row_count = len(sampled_df)
        
        # 创建输出目录
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存采样数据
        sampled_df.to_parquet(output_path, index=False)
        
        # 立即释放内存
        del df
        del sampled_df
        gc.collect()
        
        return True, actual_tokens, row_count, output_path
        
    except Exception as e:
        worker_logger.error(f"采样失败: {parquet_path}, 错误: {e}")
        import traceback
        worker_logger.error(traceback.format_exc())
        return False, 0, 0, output_path
