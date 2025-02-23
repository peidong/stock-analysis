#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
模仿通达信语句的函数库，如MA(C,5) REF(C,1)等样式。函数简单，只为了和通达信公式看起来一致，方便排查。
传入类型必须是pandas Series类型。
传出类型：只有MA输出具体数值，其他所有函数传出仍然是Series类型
作者：wking [http://wkings.net]
"""

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
        value = value.fillna(method='ffill')  # 向下填充无效值
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


def BARSLAST(series):
    # 上一次条件成立到当前的周期数.
    # 用法:
    #  BARSLAST(X):上一次X不为0到现在的天数
    # 例如:
    #  BARSLAST(CLOSE/REF(CLOSE,1)>=1.1)表示上一个涨停板到当前的周期数
    result = pd.Series(index=series.index, dtype=int)
    i = 0
    for k, v in series.iteritems():
        if v:
            i = 0
            result[k] = i
        else:
            i = i + 1
            result[k] = i
    return result


def BARSLASTCOUNT(cond):
    # 统计连续满足条件的周期数.
    # 用法:
    #  BARSLASTCOUNT(X),统计连续满足X条件的周期数.
    # 例如:
    #  BARSLASTCOUNT(CLOSE>OPEN)表示统计连续收阳的周期数
    result = pd.Series(index=cond.index, dtype=int)
    i = 0
    for k, v in cond.iteritems():
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
