# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:26:15 2023

@author: wytheY
"""

import pandas as pd

# 创建一个时间序列
date_rng = pd.date_range(start='1/1/2020', end='1/10/2020', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
s = ''
df['data'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df['data'] = pd.read_csv(r'E:/股市数据/a股日线/{}.csv'.format(s), index_col = 0, encoding='gbk')
df['data'] = df['收盘价']
# 设置日期为索引
df.set_index('date', inplace=True)

# 计算滑动窗口的平均值和标准差
window_size = 3
df['rolling_mean'] = df['data'].rolling(window=window_size).mean()
df['rolling_std'] = df['data'].rolling(window=window_size).std()

# 计算Z-Score
df['z_score'] = (df['data'] - df['rolling_mean']) / df['rolling_std']

print(df)
