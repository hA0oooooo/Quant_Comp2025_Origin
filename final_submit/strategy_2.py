import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')


def generate_signals(df):
    """
    随机策略 - 随机生成交易信号
    
    参数:
    df: 包含OHLCV数据的DataFrame
    """
    df = df.copy()  # 复制数据，避免修改原始数据
    
    # 随机生成[0, 1， -1]之间的数
    # np.random.choice()是numpy的随机选择函数
    # size=len(df)表示生成与数据行数相同数量的随机数
    # p=[0.999, 0.0005, 0.0005]表示概率分布：
    # - 0的概率是0.999（99.9%）
    # - 1的概率是0.0005（0.05%）
    # - -1的概率是0.0005（0.05%）
    df['position'] = np.random.choice([0, 1, -1], size=len(df), p=[0.999, 0.0005, 0.0005])
    ## 注意仓位的时间戳和主力合约的时间戳要对齐!
    return df['position']




# Quick example usage (assuming *df* is your high‑frequency DataFrame):
# positions = generate_signals_optimized(df, use_obi=True)
# df["position"] = positions


def pred(date):
    """
    预测函数 - 对指定日期的所有主力合约生成交易信号
    
    参数:
    date: 交易日期字符串，如'20241009'
    """
    test_dir = './future_L2/test'  # 测试数据目录
    pred_dir = './positions'       # 预测结果输出目录
    
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    Mfiles = os.listdir(f'{test_dir}/{date}')
    Mfiles = [f for f in Mfiles if '_M' in f]

    result_dict = {}

    # 对每个主力合约生成信号
    for f in Mfiles:  # for循环遍历文件名列表
        df = pd.read_parquet(f'{test_dir}/{date}/{f}')  # 读取parquet文件
        result = generate_signals(df)  # 调用随机策略生成信号
        result_dict[f.split('.')[0]] = result  # 将结果存入字典，split('.')[0]去掉文件扩展名

    # 保存结果到CSV文件
    os.makedirs(f'{pred_dir}/{date}', exist_ok=True)  # 创建目录，exist_ok=True表示目录已存在时不报错
    for code, result in result_dict.items():  # items()返回字典的键值对
        result.to_csv(f'{pred_dir}/{date}/{code}.csv')  # 保存为CSV文件



if __name__ == '__main__':

    test_dir = './future_L2/test'  # 测试数据目录
    pred_dir = './positions'       # 预测结果输出目录
    test_dates = sorted(os.listdir(test_dir))[1:]  # 获取所有测试日期，跳过第一个
    os.makedirs(pred_dir, exist_ok=True)  # 创建输出目录

    # 单进程处理版本（已注释）
    # for date in test_dates:
    #     print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    #     pred(date)

    # 多进程并行处理版本 - 提高处理速度
    from multiprocessing import Pool  # 导入多进程模块
    with Pool(20) as p:  # 创建20个进程的进程池
        p.map(pred, test_dates)  # map()将函数应用到每个日期上
