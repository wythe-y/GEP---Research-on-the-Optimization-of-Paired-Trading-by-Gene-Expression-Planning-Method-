import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller

pairs = [
    ('600519.SH', '000858.SZ'),
    ('600028.SH', '601857.SH'),
    ('002594.SZ', '601633.SH'),
    ('600887.SH', '002714.SZ'),
    ('600011.SH', '601991.SH')
]

def analyse_pair(s1, s2):
    stock1 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(s1),encoding='gbk')  
    stock2 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(s2),encoding='gbk') 

    stock1.set_index('交易时间', inplace=True)
    stock2.set_index('交易时间', inplace=True)

    aligned_stock1, aligned_stock2 = stock1.align(stock2, join='inner')

    stock1 = aligned_stock1['收盘价']
    stock2 = aligned_stock2['收盘价']

    slope, intercept, _, _, _ = linregress(stock1, stock2)
    stock2_predicted = intercept + slope * stock1
    residuals = stock2 - stock2_predicted

    result = adfuller(residuals)

    mean = np.mean(residuals)
    std = np.std(residuals)

    z_scores = (residuals - mean) / std

    print(f'For pair {s1} and {s2}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    print(f'Mean: {mean}, Std Dev: {std}')
    print('Z-Scores:', z_scores)

    residuals = residuals.to_frame()
    stock1 = stock1.to_frame()
    stock2 = stock2.to_frame()

    df = pd.concat([stock1, stock2, residuals, z_scores], axis=1)
    df.columns = [f'{s1}_Close', f'{s2}_Close', 'Residual', 'Z-Score']

    print(df)
    df.plot(figsize=(8,8))

for pair in pairs:
    analyse_pair(*pair)
