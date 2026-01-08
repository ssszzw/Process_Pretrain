"""
计算单个 parquet 文件的 token 数量
使用多进程并行处理
"""
import gc
import pandas as pd
from transformers import AutoTokenizer
import logging
from typing import Tuple, Optional, List, Union

# 配置日志
logger = logging.getLogger(__name__)


def count_tokens_in_file(args) -> Tuple[str, int, int, Optional[str]]:
    """
    计算单个 parquet 文件中的 token 数量
    
    Args:
        args: (file_path, tokenizer_path, text_columns, batch_size, worker_id)
              text_columns 可以是字符串或列表，如果是列表则依次尝试查找
        
    Returns:
        (file_path, token_count, row_count, error_message)
        如果成功则 error_message 为 None
    """
    file_path, tokenizer_path, text_columns, batch_size, worker_id = args
    
    # 为每个进程配置独立的 logger
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    
    # 标准化 text_columns 为列表
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    
    try:
        # 加载 tokenizer（每个进程加载一次）
        if not hasattr(count_tokens_in_file, f'tokenizer_{worker_id}'):
            worker_logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            setattr(count_tokens_in_file, f'tokenizer_{worker_id}', tokenizer)
        else:
            tokenizer = getattr(count_tokens_in_file, f'tokenizer_{worker_id}')
        
        worker_logger.info(f"Processing: {file_path}")
        
        # 首先读取文件获取所有列信息
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            error_msg = f"Failed to read file: {str(e)}"
            worker_logger.error(f"{file_path}: {error_msg}")
            gc.collect()
            return file_path, 0, 0, error_msg
        
        # 依次尝试查找指定的列名
        text_column = None
        for col_name in text_columns:
            if col_name in df.columns:
                text_column = col_name
                worker_logger.info(f"Found text column: '{text_column}'")
                break
        
        # 如果没有找到任何指定的列
        if text_column is None:
            available_cols = ', '.join(df.columns.tolist())
            tried_cols = ', '.join(text_columns)
            error_msg = f"None of the specified columns [{tried_cols}] found. Available columns: {available_cols}..."
            worker_logger.error(f"{file_path}: {error_msg}")
            del df
            gc.collect()
            return file_path, 0, 0, error_msg
        
        row_count = len(df)
        total_tokens = 0
        
        if row_count == 0:
            worker_logger.warning(f"Empty file: {file_path}")
            return file_path, 0, 0, None
        
        # 批处理以避免内存问题
        for i in range(0, row_count, batch_size):
            batch = df[text_column].iloc[i:i+batch_size]
            
            # 过滤掉空值并转换为字符串
            texts = []
            for text in batch:
                if pd.notna(text) and text:
                    texts.append(str(text))
            
            if not texts:
                continue
            
            # Tokenize 并计数
            try:
                tokens = tokenizer(
                    texts,
                    truncation=False,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )
                
                # 计算这批数据的总 token 数
                for input_ids in tokens['input_ids']:
                    total_tokens += len(input_ids)
                
                # 清理批次内存
                del tokens
                    
            except Exception as e:
                worker_logger.warning(f"Tokenization error in batch: {e}")
                continue
        
        worker_logger.info(f"Completed: {file_path} - {total_tokens:,} tokens, {row_count:,} rows")
        
        # 清理内存
        del df
        gc.collect()
        
        return file_path, total_tokens, row_count, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        worker_logger.error(f"Failed to process {file_path}: {error_msg}")
        # 确保异常情况下也清理内存
        gc.collect()
        return file_path, 0, 0, error_msg
