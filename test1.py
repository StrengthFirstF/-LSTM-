import pandas as pd

# 步骤1：读取CSV文件并解析时间列
def sort_csv_by_time(input_file, output_file, time_column):
    """
    参数说明：
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    time_column: 时间列的名称
    """
    try:
        # 读取CSV并自动解析时间列
        df = pd.read_csv(
            input_file,
            parse_dates=[time_column],  # 自动解析时间列
            infer_datetime_format=True,  # 自动推断时间格式
            encoding='utf-8'            # 根据文件实际编码调整
        )
        
        # 步骤2：验证时间列是否存在
        if time_column not in df.columns:
            raise ValueError(f"时间列'{time_column}'不存在于文件中")
        
        # 步骤3：按时间排序（默认升序）
        sorted_df = df.sort_values(by=time_column)
        
        # 步骤4：保存排序后的结果
        sorted_df.to_csv(
            output_file,
            index=False,               # 不保存行索引
            encoding='utf-8-sig'       # 支持中文的编码格式
        )
        
        print(f"文件已按 [{time_column}] 排序并保存至: {output_file}")
        print(f"原始数据量: {len(df)} 条")
        print(f"排序后数据量: {len(sorted_df)} 条")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 设置文件路径参数
    input_csv = "train_data.csv"    # 输入文件路径
    output_csv = "train_data_sort.csv" # 输出文件路径
    time_col = "数据时间"         # 时间列名称（需要与CSV列名完全一致）
    
    # 执行排序
    sort_csv_by_time(input_csv, output_csv, time_col)