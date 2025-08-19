import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')


def generate_signals(df, window=3600):
    """
    基础突破策略 - 简单的价格突破+成交量确认策略
    
    参数:
    df: 包含OHLCV数据的DataFrame
    window: 滚动窗口大小，默认3600个tick（约1小时）
    """
    df = df.copy()
    
    # 计算动态区间 - 滚动窗口计算最高价、最低价和成交量均值
    df['rolling_high'] = df['HIGHPRICE'].shift(1).rolling(window).max()  # 前window个周期的最高价
    df['rolling_low'] = df['LOWPRICE'].shift(1).rolling(window).min()    # 前window个周期的最低价
    df['volume_ma'] = df['TRADEVOLUME'].rolling(window).mean()           # 成交量移动平均

    # 突破信号判断 - 价格突破极值且继续朝突破方向移动
    df['break_high'] = (df['LASTPRICE'] > df['rolling_high']) & (df['LASTPRICE'] > df['LASTPRICE'].shift(1))  # 向上突破且价格上涨
    df['break_low'] = (df['LASTPRICE'] < df['rolling_low']) & (df['LASTPRICE'] < df['LASTPRICE'].shift(1))    # 向下突破且价格下跌

    # 有效突破 - 需要放量确认，成交量大于均值的1.2倍
    df['valid_break_high'] = df['break_high'] & (df['TRADEVOLUME'] > 1.2 * df['volume_ma'])  # 成交量大于均值的1.2倍
    df['valid_break_low'] = df['break_low'] & (df['TRADEVOLUME'] > 1.2 * df['volume_ma'])
    
    # 初始化持仓列
    df['position'] = 0

    # 注释掉的简单版本：直接根据突破信号设置仓位
    # df['position'] = np.where(df['break_high'], 1, np.where(df['break_low'], -1, 0))
    
    # 手动实现仓位管理逻辑，确保仓位状态的一致性
    current_position = 0  # 当前持仓状态：0=空仓，1=多仓，-1=空仓
    entry_price = 0      # 入场价格

    # 逐tick遍历，实现仓位管理
    for i in range(len(df)):
        # 空仓时检测突破信号
        if current_position == 0:
            if df['valid_break_high'].iloc[i]:      # 向上突破，开多仓
                current_position = 1
                entry_price = df['LASTPRICE'].iloc[i]
            elif df['valid_break_low'].iloc[i]:     # 向下突破，开空仓
                current_position = -1
                entry_price = df['LASTPRICE'].iloc[i]
        
        # 持多仓时检测平仓条件
        elif current_position == 1:
            if (df['LASTPRICE'].iloc[i] >= entry_price * 1.003) or \
               (df['LASTPRICE'].iloc[i] <= entry_price * 0.999):      # 止盈：上涨0.3% 或 止损：下跌0.1%
                current_position = 0
                
        # 持空仓时检测平仓条件
        elif current_position == -1:
            # 止盈：下跌0.3% 或 止损：上涨0.1%
            if (df['LASTPRICE'].iloc[i] <= entry_price * 0.997) or \
               (df['LASTPRICE'].iloc[i] >= entry_price * 1.001):      # 止盈：下跌0.3% 或 止损：上涨0.1%
                current_position = 0
        
        df['position'].iloc[i] = current_position  # 将当前仓位状态保存到DataFrame中
    
    
    ## 注意仓位的时间戳和主力合约的时间戳要对齐!
    return df['position']


def generate_signals_optimized(
    df: pd.DataFrame,

    window: int = 7200,                 # 滚动窗口大小，默认7200个tick
    ema_period: int = 1080,             # EMA周期，默认1080个tick
    vol_multiplier: float = 1.8,        # 成交量倍数，默认1.8倍
    atr_period: int = 240,              # ATR周期，默认240个tick; Average True Range（平均真实波幅）
    stop_loss_mult: float = 1.0,        # 止损倍数，默认1.0倍ATR
    take_profit_mult: float = 3.0,      # 止盈倍数，默认3.0倍ATR
    time_stop: int = 21600,             # 时间止损，默认21600个tick
    use_obi: bool = True,               # 是否使用盘口不均衡指标
    obi_threshold: float = 0.15,        # 盘口不均衡阈值

    bid_col: str = "BUYVOLUME01",       # 买一量列名
    ask_col: str = "SELLVOLUME01",      # 卖一量列名
    high_col: str = "HIGHPRICE",        # 最高价列名
    low_col: str = "LOWPRICE",          # 最低价列名
    close_col: str = "LASTPRICE",       # 收盘价列名
    vol_col: str = "TRADEVOLUME",       # 成交量列名
) -> pd.Series:  # 返回Series类型
    """
    优化突破策略 - 高频突破策略，包含趋势/VWAP/盘口不均衡过滤和ATR风险控制
    
    主要改进：
    1. 多重技术指标过滤（EMA、VWAP、OBI）
    2. ATR动态止损和止盈
    3. 移动止损机制
    4. 时间止损
    
    Returns
    -------
    pd.Series
        Position series aligned with *df* (1 = long, −1 = short, 0 = flat).
    """
    data = df.copy()

    # ─────────────────────────────
    # 0. VWAP计算 - 成交量加权平均价格，用于判断价格相对强弱
    # cumsum()计算累积和，replace(0, np.nan)将0替换为NaN
    cum_vol = data[vol_col].cumsum().replace(0, np.nan)
    # 计算VWAP：价格乘以成交量的累积和除以成交量的累积和
    data["vwap"] = (data[close_col] * data[vol_col]).cumsum() / cum_vol

    # 1. 滚动极值和成交量参考 - 计算动态支撑阻力位
    data["rolling_high"] = data[high_col].shift().rolling(window).max()  # 滚动最高价
    data["rolling_low"]  = data[low_col].shift().rolling(window).min()   # 滚动最低价
    data["vol_ma"]        = data[vol_col].rolling(window).mean()         # 成交量均值

    # 2. 趋势过滤器 - 指数移动平均，过滤逆势信号
    # ewm()计算指数加权移动平均，span是平滑因子
    data["ema"] = data[close_col].ewm(span=ema_period, adjust=False).mean()

    # 3. ATR计算 - 平均真实波幅，用于动态止损止盈
    high, low, close = data[high_col], data[low_col], data[close_col]
    true_range = pd.concat([
        high - low,                                    # 当日高低价差
        (high - close.shift()).abs(),                 # 当日最高价与前收盘价差
        (low  - close.shift()).abs(),                 # 当日最低价与前收盘价差
    ], axis=1).max(axis=1)  # max(axis=1)取每行的最大值
    data["atr"] = true_range.rolling(atr_period).mean()  # ATR 是真实波幅的移动平均

    # 4. 盘口不均衡指标 - 买卖盘力量对比，过滤弱势突破
    if use_obi:
        data["obi"] = (data[bid_col] - data[ask_col]) / (
            data[bid_col] + data[ask_col] + 1e-12
        )
    else:
        data["obi"] = 0.0  # dummy

    # 5. 突破条件 - 多重过滤，提高信号质量
    # 多头条件：价格突破高点 + 放量确认 + 趋势向上 + 价格在VWAP之上 + 买盘力量强
    cond_long = (
        (close > data["rolling_high"]) &              # 价格突破高点
        (data[vol_col] > vol_multiplier * data["vol_ma"]) &  # 放量确认
        (close > data["ema"]) &                       # 趋势向上
        (close > data["vwap"]) &                      # 价格在VWAP之上
        (~use_obi | (data["obi"] >  obi_threshold))   # 买盘力量强
    )

    # 空头条件：价格突破低点 + 放量确认 + 趋势向下 + 价格在VWAP之下 + 卖盘力量强
    cond_short = (
        (close < data["rolling_low"]) &               # 价格突破低点
        (data[vol_col] > vol_multiplier * data["vol_ma"]) &  # 放量确认
        (close < data["ema"]) &                       # 趋势向下
        (close < data["vwap"]) &                      # 价格在VWAP之下
        (~use_obi | (data["obi"] < -obi_threshold))   # 卖盘力量强
    )

    data["break_high"], data["break_low"] = cond_long, cond_short

    # 6. 仓位管理 - 使用ATR跟踪止损和动态止盈
    pos = np.zeros(len(data), dtype=np.int8)
    current_pos, entry_price, entry_idx = 0, 0.0, -1
    peak, trough = 0.0, 0.0  # 跟踪极值用于移动止损

    for i, (price, atr_val) in enumerate(zip(close.to_numpy(), data["atr"].to_numpy())):
        if not np.isfinite(price):  
            pos[i] = current_pos
            continue

        if current_pos == 0:  # 空仓时寻找入场机会
            if data["break_high"].iat[i]:
                current_pos = 1
                entry_price, entry_idx, peak = price, i, price
            elif data["break_low"].iat[i]:
                current_pos = -1
                entry_price, entry_idx, trough = price, i, price

        elif current_pos == 1:  # 多仓管理
            peak = max(peak, price)  # 更新最高点
            # 移动止损：取入场止损和跟踪止损的最大值
            sl = max(entry_price - stop_loss_mult * atr_val, peak - stop_loss_mult * atr_val)
            tp = entry_price + take_profit_mult * atr_val  # 止盈
            # 止损/止盈/时间止损
            if price <= sl or price >= tp or i - entry_idx >= time_stop:
                current_pos = 0

        elif current_pos == -1:  # 空仓管理
            trough = min(trough, price)  # 更新最低点
            # 移动止损：取入场止损和跟踪止损的最小值
            sl = min(entry_price + stop_loss_mult * atr_val, trough + stop_loss_mult * atr_val)
            tp = entry_price - take_profit_mult * atr_val  # 止盈
            # 止损/止盈/时间止损
            if price >= sl or price <= tp or i - entry_idx >= time_stop:
                current_pos = 0

        pos[i] = current_pos

    # 返回持仓序列，ffill()向前填充，保证连续仓位，fillna(0)将NaN填充为0
    return pd.Series(pos, index=data.index, name="position").ffill().fillna(0)


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
    Mfiles = [f for f in Mfiles if '_M' in f]  # 只处理主力合约

    result_dict = {}

    # 对每个主力合约生成信号
    for f in Mfiles:
        df = pd.read_parquet(f'{test_dir}/{date}/{f}')
        # result = generate_signals(df)  # 使用基础策略
        result = generate_signals_optimized(df)  # 使用优化策略
        result_dict[f.split('.')[0]] = result

    # 保存结果到CSV文件
    os.makedirs(f'{pred_dir}/{date}', exist_ok=True)  # 创建目录，exist_ok=True表示目录已存在时不报错
    for code, result in result_dict.items():  # items()返回字典的键值对
        result.to_csv(f'{pred_dir}/{date}/{code}.csv')  # 保存为CSV文件





if __name__ == '__main__':

    test_dir = './future_L2/test'  # 测试数据目录
    pred_dir = './positions'        # 预测结果输出目录
    test_dates = sorted(os.listdir(test_dir))[1:]  # 获取所有测试日期，跳过第一个
    os.makedirs(pred_dir, exist_ok=True)

    # 单进程处理版本（已注释）
    # for date in test_dates:
    #     print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    #     pred(date)

    # 多进程并行处理版本 - 提高处理速度
    from multiprocessing import Pool  # 导入多进程模块
    with Pool(20) as p:  # 创建20个进程的进程池
        p.map(pred, test_dates)  # map()将函数应用到每个日期上
