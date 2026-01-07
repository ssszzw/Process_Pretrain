from pathlib import Path

def get_relative_path(file_path, source_dir):
    """获取相对于源目录的相对路径"""
    return Path(file_path).relative_to(source_dir)


def create_target_path(relative_path, target_dir):
    """创建目标文件路径"""
    target_path = Path(target_dir) / relative_path
    # 创建目标目录
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path
