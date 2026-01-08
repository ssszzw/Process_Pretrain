"""
Token 统计结果分析工具
用于分析和可视化 token 计数结果
"""
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def load_results(json_file: str) -> Dict:
    """加载结果文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_number(num: int) -> str:
    """格式化大数字"""
    if num >= 1_000_000_000_000:  # T
        return f"{num/1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:  # B
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:  # M
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:  # K
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def print_summary(results: Dict):
    """打印总体摘要"""
    print("=" * 100)
    print("TOKEN 统计汇总")
    print("=" * 100)
    print()
    
    # 总体统计
    print("总体统计:")
    print(f"  总 Token 数:        {results['total_tokens']:>20,} ({format_number(results['total_tokens'])})")
    print(f"  总行数:             {results['total_rows']:>20,} ({format_number(results['total_rows'])})")
    print(f"  总文件数:           {results['total_files']:>20,}")
    print(f"  失败文件数:         {results['failed_files']:>20,}")
    
    if results['total_rows'] > 0:
        avg_tokens = results['total_tokens'] / results['total_rows']
        print(f"  平均 Tokens/行:     {avg_tokens:>20,.2f}")
    print()
    
    # 元数据
    if 'metadata' in results:
        meta = results['metadata']
        print("处理信息:")
        print(f"  数据根目录:         {meta.get('data_root', 'N/A')}")
        print(f"  Tokenizer:          {meta.get('tokenizer_path', 'N/A')}")
        print(f"  处理时间:           {meta.get('timestamp', 'N/A')}")
        print(f"  耗时:               {meta.get('processing_time_seconds', 0):.2f} 秒")
        print(f"  节点数:             {meta.get('num_nodes', 'N/A')}")
        print(f"  每节点进程数:       {meta.get('num_workers_per_node', 'N/A')}")
        
        # 计算处理速度
        if meta.get('processing_time_seconds', 0) > 0:
            tokens_per_sec = results['total_tokens'] / meta['processing_time_seconds']
            print(f"  处理速度:           {format_number(int(tokens_per_sec))}/秒")
        print()


def print_dataset_details(results: Dict, sort_by: str = 'tokens', top_n: Optional[int] = None):
    """打印数据集详情"""
    print("=" * 100)
    print("数据集详细信息")
    print("=" * 100)
    print()
    
    datasets = results.get('datasets', {})
    
    # 排序
    if sort_by == 'tokens':
        sorted_datasets = sorted(
            datasets.items(),
            key=lambda x: x[1]['total_tokens'],
            reverse=True
        )
    elif sort_by == 'name':
        sorted_datasets = sorted(datasets.items())
    else:
        sorted_datasets = list(datasets.items())
    
    # 限制显示数量
    if top_n:
        sorted_datasets = sorted_datasets[:top_n]
    
    # 打印每个数据集
    for idx, (dataset_name, dataset_stats) in enumerate(sorted_datasets, 1):
        print(f"{idx}. 数据集: {dataset_name}")
        print(f"   总 Token 数:       {dataset_stats['total_tokens']:>20,} ({format_number(dataset_stats['total_tokens'])})")
        print(f"   总行数:            {dataset_stats['total_rows']:>20,} ({format_number(dataset_stats['total_rows'])})")
        print(f"   总文件数:          {dataset_stats['total_files']:>20,}")
        print(f"   失败文件数:        {dataset_stats['failed_files']:>20,}")
        
        # 计算占比
        if results['total_tokens'] > 0:
            percentage = (dataset_stats['total_tokens'] / results['total_tokens']) * 100
            print(f"   占比:              {percentage:>20.2f}%")
        
        # 平均 tokens/行
        if dataset_stats['total_rows'] > 0:
            avg_tokens = dataset_stats['total_tokens'] / dataset_stats['total_rows']
            print(f"   平均 Tokens/行:    {avg_tokens:>20,.2f}")
        
        # 打印子集信息
        subsets = dataset_stats.get('subsets', {})
        if subsets:
            print(f"   子集数量: {len(subsets)}")
            
            # 按 token 数量排序子集
            sorted_subsets = sorted(
                subsets.items(),
                key=lambda x: x[1]['token_count'],
                reverse=True
            )
            
            for subset_name, subset_stats in sorted_subsets:
                subset_pct = (subset_stats['token_count'] / dataset_stats['total_tokens'] * 100) \
                    if dataset_stats['total_tokens'] > 0 else 0
                
                print(f"     - {subset_name}:")
                print(f"         Tokens:      {subset_stats['token_count']:>15,} ({format_number(subset_stats['token_count'])}) [{subset_pct:.1f}%]")
                print(f"         行数:        {subset_stats['row_count']:>15,}")
                print(f"         文件数:      {subset_stats['file_count']:>15,}")
                if subset_stats['failed_files'] > 0:
                    print(f"         失败:        {subset_stats['failed_files']:>15,}")
        
        print()


def generate_distribution_chart(results: Dict, top_n: int = 30):
    """生成分布图（文本版）"""
    print("=" * 100)
    print(f"TOP {top_n} 数据集（按 Token 数量）")
    print("=" * 100)
    print()
    
    datasets = results.get('datasets', {})
    
    # 按 token 数量排序
    sorted_datasets = sorted(
        datasets.items(),
        key=lambda x: x[1]['total_tokens'],
        reverse=True
    )[:top_n]
    
    if not sorted_datasets:
        print("没有数据")
        return
    
    # 找出最大值用于归一化
    max_tokens = sorted_datasets[0][1]['total_tokens']
    
    # 打印条形图
    bar_width = 60
    for i, (dataset_name, stats) in enumerate(sorted_datasets, 1):
        tokens = stats['total_tokens']
        bar_length = int((tokens / max_tokens) * bar_width) if max_tokens > 0 else 0
        bar = "█" * bar_length
        
        # 截断名称
        display_name = dataset_name[:35]
        
        # 计算占比
        percentage = (tokens / results['total_tokens'] * 100) if results['total_tokens'] > 0 else 0
        
        print(f"{i:2}. {display_name:<35} {bar} {format_number(tokens):>8} ({percentage:5.1f}%)")
    
    print()


def generate_subset_distribution_chart(results: Dict, top_n: int = 50):
    """
    生成子集分布图（文本版）
    展示所有子集的 token 量分布
    
    Args:
        results: 统计结果字典
        top_n: 显示前 N 个子集（默认 50）
    """
    print("=" * 100)
    print(f"TOP {top_n} 子集（按 Token 数量）")
    print("=" * 100)
    print()
    
    datasets = results.get('datasets', {})
    
    # 收集所有子集信息
    all_subsets = []
    for dataset_name, dataset_stats in datasets.items():
        subsets = dataset_stats.get('subsets', {})
        for subset_name, subset_stats in subsets.items():
            all_subsets.append({
                'dataset': dataset_name,
                'subset': subset_name,
                'tokens': subset_stats['token_count'],
                'rows': subset_stats['row_count'],
                'files': subset_stats['file_count'],
                'failed': subset_stats['failed_files']
            })
    
    if not all_subsets:
        print("没有子集数据")
        return
    
    # 按 token 数量排序
    sorted_subsets = sorted(all_subsets, key=lambda x: x['tokens'], reverse=True)[:top_n]
    
    # 找出最大值用于归一化
    max_tokens = sorted_subsets[0]['tokens']
    
    # 打印条形图
    bar_width = 50
    for i, subset_info in enumerate(sorted_subsets, 1):
        tokens = subset_info['tokens']
        bar_length = int((tokens / max_tokens) * bar_width) if max_tokens > 0 else 0
        bar = "█" * bar_length
        
        # 构建显示名称: dataset/subset
        full_name = f"{subset_info['dataset']}/{subset_info['subset']}"
        display_name = full_name[:60]  # 截断名称
        
        # 计算占比
        percentage = (tokens / results['total_tokens'] * 100) if results['total_tokens'] > 0 else 0
        
        # 显示信息
        print(f"{i:3}. {display_name:<40} {bar} {format_number(tokens):>8} ({percentage:5.2f}%)")
    
    print()
    
    # 打印统计信息
    print(f"子集总数: {len(all_subsets)}")
    print(f"显示数量: {len(sorted_subsets)}")
    
    # 计算前 N 个子集的总占比
    top_n_tokens = sum(s['tokens'] for s in sorted_subsets)
    top_n_percentage = (top_n_tokens / results['total_tokens'] * 100) if results['total_tokens'] > 0 else 0
    print(f"前 {len(sorted_subsets)} 个子集占总 Token 数: {top_n_percentage:.2f}%")
    print()


def export_to_csv(results: Dict, output_csv: str):
    """导出为 CSV 格式"""
    import csv
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            'Dataset', 'Subset', 'Token Count', 'Row Count', 
            'File Count', 'Failed Files', 'Percentage', 'Avg Tokens/Row'
        ])
        
        # 写入数据
        datasets = results.get('datasets', {})
        for dataset_name, dataset_stats in sorted(datasets.items()):
            subsets = dataset_stats.get('subsets', {})
            
            for subset_name, subset_stats in sorted(subsets.items()):
                percentage = (subset_stats['token_count'] / results['total_tokens'] * 100) \
                    if results['total_tokens'] > 0 else 0
                
                avg_tokens = (subset_stats['token_count'] / subset_stats['row_count']) \
                    if subset_stats['row_count'] > 0 else 0
                
                writer.writerow([
                    dataset_name,
                    subset_name,
                    subset_stats['token_count'],
                    subset_stats['row_count'],
                    subset_stats['file_count'],
                    subset_stats['failed_files'],
                    f"{percentage:.4f}",
                    f"{avg_tokens:.2f}"
                ])
    
    print(f"结果已导出到: {output_csv}")


def print_usage():
    """打印使用说明"""
    print("使用方法:")
    print("  python analyze_results.py <results.json> [options]")
    print()
    print("选项:")
    print("  --csv <output.csv>     导出为 CSV 格式")
    print("  --top N                只显示前 N 个数据集（默认显示全部）")
    print("  --sort [tokens|name]   排序方式（默认按 tokens）")
    print()
    print("示例:")
    print("  python analyze_results.py token_statistics.json")
    print("  python analyze_results.py token_statistics.json --csv output.csv")
    print("  python analyze_results.py token_statistics.json --top 10")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        sys.exit(1)
    
    # 解析参数
    sort_by = 'tokens'
    top_n = None
    csv_output = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--csv' and i + 1 < len(sys.argv):
            csv_output = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--top' and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--sort' and i + 1 < len(sys.argv):
            sort_by = sys.argv[i + 1]
            i += 2
        else:
            print(f"未知选项: {sys.argv[i]}")
            print_usage()
            sys.exit(1)
    
    # 加载结果
    print(f"加载结果: {json_file}")
    print()
    results = load_results(json_file)
    
    # 打印摘要
    print_summary(results)
    
    # 打印数据集详情
    print_dataset_details(results, sort_by=sort_by, top_n=top_n)
    
    # 打印数据集分布图
    generate_distribution_chart(results, top_n=20)
    
    # 打印子集分布图
    generate_subset_distribution_chart(results, top_n=50)
    
    # 导出 CSV（如果指定）
    if csv_output:
        export_to_csv(results, csv_output)
        print()
    
    print("=" * 100)
    print("分析完成!")
    print("=" * 100)


if __name__ == "__main__":
    main()
