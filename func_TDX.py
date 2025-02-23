#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
模仿通达信语句的函数库，如MA(C,5) REF(C,1)等样式。函数简单，只为了和通达信公式看起来一致，方便排查。
传入类型必须是pandas Series类型。
传出类型：只有MA输出具体数值，其他所有函数传出仍然是Series类型
作者：wking [http://wkings.net]
"""
import numpy as np
import pandas as pd


def rolling_window(a, window):
    """
    copy from http://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
    必须 numpy > 1.20 才有此函数
    """
    # 从 numpy 的 lib.stride_tricks 模块中导入 sliding_window_view 函数
    from numpy.lib.stride_tricks import sliding_window_view
    # 使用 sliding_window_view 函数对输入数组 a 进行滑动窗口操作
    # window_shape 参数指定窗口的大小，这里使用传入的 window 参数
    return sliding_window_view(a, window_shape=window)


def REF(value, day):
    """
    引用若干周期前的数据。如果传入列表，返回具体数值。如果传入序列，返回序列
    """
    # 检查传入的value的类型是否为列表
    if 'list' in str(type(value)):
        # 如果是列表，使用负索引获取day周期前的数据
        result = value[~day]
    # 检查传入的value的类型是否为序列（假设为pandas.Series）
    elif 'series' in str(type(value)):
        # 如果是序列，使用shift方法获取day周期前的数据
        result = value.shift(periods=day)
    # 返回结果
    return result


def MA(value, day) -> float:
    """
    返回当前周期的简单移动平均值。传入可以是列表或序列类型。传出是当前周期的简单移动平均具体值。
    :rtype: float
    """
    import talib  # 导入talib库，用于金融技术分析
    # result = statistics.mean(value[-day:])
    result = talib.SMA(value, day).iat[-1]
    return result


def SMA(value, day):
    """
    返回简单移动平均序列。传入可以是列表或序列类型。传出是历史到当前周期为止的简单移动平均序列。
    """
    import talib
    # result = statistics.mean(value[-day:])
    result = talib.SMA(value, day)
    return result


def HHV(series, day):
    """
    返回最大值
    """
    # value = max(series[-day:])
    if day == 0:
        value = pd.Series(index=series.index, dtype=float)
        tmp = series.iat[0]
        value.iat[0] = tmp
        for i in range(series.shape[0]):
            if tmp < series.iat[i]:
                tmp = series.iat[i]
                value.iat[i] = tmp
        value = value.fillna(method='ffill')  # 向下填充无效值
    else:
        value = series.rolling(day).max()
        value.iloc[0:day-1] = HHV(series.iloc[0:day-1], 0)
    return value


def LLV(series, day):
    """
    返回最小值
    """
    # value = min(value[-day:])
    if day == 0:
        value = pd.Series(index=series.index, dtype=float)
        tmp = series.iat[0]
        value.iat[0] = tmp
        for i in range(series.shape[0]):
            if tmp > series.iat[i]:
                tmp = series.iat[i]
                value.iat[i] = tmp
        # value = value.fillna(method='ffill')  # 向下填充无效值
        value = value.ffill()
    else:
        value = series.rolling(day).min()
        value.iloc[0:day - 1] = LLV(series.iloc[0:day - 1], 0)
    return value


def COUNT(series, n):
    # rolling方法不行，虽然简单明了但是性能太差
    # result = series.rolling(n) \
    #     .apply(lambda x: x.value_counts().to_dict()[True] if True in x.value_counts().to_dict() else 0)
    df = series.to_frame('cond')
    df.insert(df.shape[1], 'result', 0)
    for index_true in df.loc[df['cond'] == True].index.to_list():
        index_int = df.index.get_loc(index_true)
        column_int = df.columns.get_loc('result')
        df.iloc[index_int:index_int + n, column_int] = df.iloc[index_int:index_int + n, column_int] + 1
    result = df['result']
    return result


def EXIST(cond, n):
    series = cond[-n:]
    if True in series.to_list():
        return True
    else:
        return False


def CROSS(s1, s2):
    cond1 = s1 > s2
    cond2 = s1.shift() <= s2.shift()
    result = cond1 & cond2
    return result


# def BARSLAST(series):
#     # 上一次条件成立到当前的周期数.
#     # 用法:
#     #  BARSLAST(X):上一次X不为0到现在的天数
#     # 例如:
#     #  BARSLAST(CLOSE/REF(CLOSE,1)>=1.1)表示上一个涨停板到当前的周期数
#     result = pd.Series(index=series.index, dtype=int)
#     i = 0
#     # for k, v in series.iteritems():
#     for k, v in series.items():
#         if v:
#             i = 0
#             result[k] = i
#         else:
#             i = i + 1
#             result[k] = i
#     return result

def BARSLAST(series):
    """
    计算上一次条件成立到当前位置的周期数。
    :param series: 布尔序列或可转换为布尔值的序列
    :return: 周期数序列，未记录时返回-1
    """
    # 将输入转换为布尔序列，忽略NaN和非布尔值
    bool_series = series.astype(bool).fillna(False)
    result = pd.Series(index=series.index, dtype=int)
    last_valid = -1  # 上一次条件成立的位置
    for i, v in enumerate(bool_series):
        if v:
            last_valid = i
            # result[i] = 0
            result.iloc[i] = 0
        else:
            if last_valid != -1:
                # result[i] = i - last_valid
                result.iloc[i] = i - last_valid
            else:
                # result[i] = -1
                result.iloc[i] = -1
    return result


def BARSLASTCOUNT(cond):
    # 统计连续满足条件的周期数.
    # 用法:
    #  BARSLASTCOUNT(X),统计连续满足X条件的周期数.
    # 例如:
    #  BARSLASTCOUNT(CLOSE>OPEN)表示统计连续收阳的周期数
    result = pd.Series(index=cond.index, dtype=int)
    i = 0
    # for k, v in cond.iteritems():
    for k, v in cond.items():
        if v:
            i = i + 1
            result[k] = i
        else:
            i = 0
            result[k] = i
    return result


def VALUEWHEN(cond, value_series):
    result = pd.Series(index=cond.index, dtype=float)
    result.loc[cond.loc[cond==True].keys()] = value_series.loc[cond.loc[cond==True].keys()]
    result = result.fillna(method='ffill')  # 向下填充无效值
    return result


#以下是新增函数
def HIGH(series) -> pd.Series:
    """返回该周期最高价"""
    return series.max(axis=1)

def LOW(series) -> pd.Series:
    """返回该周期最低价"""
    return series.min(axis=1)

def OPEN(series) -> pd.Series:
    """返回该周期开盘价"""
    return series['open']

def CLOSE(series) -> pd.Series:
    """返回该周期收盘价"""
    return series['close']

def VOL(series) -> pd.Series:
    """返回该周期成交量"""
    return series['vol']

def AMOUNT(series) -> pd.Series:
    """返回该周期成交额"""
    return series['amount']

def ADVANCE(series) -> pd.Series:
    """返回该周期上涨家数（仅大盘有效）"""
    return series['advance']

def DECLINE(series) -> pd.Series:
    """返回该周期下跌家数（仅大盘有效）"""
    return series['decline']

def BUYVOL(series) -> pd.Series:
    """返回主动性买单量（仅个股分笔有效）"""
    return series['buyvol']

def SELLVOL(series) -> pd.Series:
    """返回主动性卖单量（仅个股分笔有效）"""
    return series['sellvol']

def ISBUYORDER(series) -> pd.Series:
    """判断该成交是否为主动性买单"""
    return series['isbuyorder']

def ISSELLORDER(series) -> pd.Series:
    """判断该成交是否为主动性卖单"""
    return series['issellorder']

def DATE(series) -> pd.Series:
    """返回该周期日期（格式：YYYYMMDD）"""
    return pd.to_datetime(series.index).dt.strftime('%Y%m%d')

def TIME(series) -> pd.Series:
    """返回该周期时分秒（格式：HHMMSS）"""
    return pd.to_datetime(series.index).dt.strftime('%H%M%S')

def YEAR(series) -> pd.Series:
    """返回该周期年份"""
    return pd.to_datetime(series.index).dt.year

def MONTH(series) -> pd.Series:
    """返回该周期月份"""
    return pd.to_datetime(series.index).dt.month

def DAY(series) -> pd.Series:
    """返回该周期日期（1-31）"""
    return pd.to_datetime(series.index).dt.day

def HOUR(series) -> pd.Series:
    """返回该周期小时（0-23）"""
    return pd.to_datetime(series.index).dt.hour

def MINUTE(series) -> pd.Series:
    """返回该周期分钟（0-59）"""
    return pd.to_datetime(series.index).dt.minute

def FROMOPEN(series) -> pd.Series:
    """返回当前时刻距开盘分钟数"""
    return (pd.to_datetime('now') - pd.to_datetime(series.index)).dt.total_seconds() // 60

def DRAWNULL(series) -> pd.Series:
    """返回无效数"""
    return pd.Series([np.nan] * len(series))

def BACKSET(series, condition, n):
    """向前赋值，将当前位置到n周期前的数据设为1"""
    result = pd.Series(index=series.index, dtype=int)
    for i in range(len(series)):
        if condition[i]:
            result[i] = 1
            for j in range(1, n + 1):
                if i - j >= 0:
                    result[i - j] = 1
        else:
            result[i] = 0
    return result

def BARSCOUNT(series) -> pd.Series:
    """返回有效数据周期数"""
    return pd.Series([len(series)] * len(series))

def CURRBARCOUNT(series) -> pd.Series:
    """返回到最终交易日的周期数"""
    return pd.Series([len(series)] * len(series))

def TOTALBARCOUNT(series) -> pd.Series:
    """返回总周期数"""
    return pd.Series([len(series)] * len(series))

def COUNT(series, condition, n):
    """统计满足条件的周期数"""
    return series.rolling(n).apply(lambda x: (x == True).sum(), raw=True)

def FILTER(series, condition, n):
    """过滤连续信号"""
    result = pd.Series(index=series.index, dtype=bool)
    flag = False
    for i in range(len(series)):
        if condition[i]:
            flag = True
            result[i] = True
        else:
            if flag:
                flag = False
                result[i] = False
            else:
                result[i] = True
    return result

def SUM(series, n):
    """计算总和"""
    return series.rolling(n).sum()

def SUMBARS(series, target):
    """累加到指定值"""
    result = pd.Series(index=series.index, dtype=int)
    current_sum = 0
    for i in range(len(series)):
        current_sum += series[i]
        if current_sum >= target:
            result[i] = i - SUMBARS(series.iloc[:i], target)
            break
    return result