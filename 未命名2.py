# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:07:05 2023

@author: wytheY
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

pairs = [
    ('600584.SH', '600730.SH'),
    # ('600998.SH', '600730.SH'),
    # ('600998.SH', '600584.SH'),
    # ('300146.SZ', '600730.SH'),
    # ('300146.SZ', '600998.SH')
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

    df = pd.DataFrame({
        s1: stock1,
        s2: stock2
    })
    
    johansen_test = coint_johansen(df, det_order=0, k_ar_diff=1)
    print(slope,intercept)
    print(f'For pair {s1} and {s2}:')
    print('Eigenvalues:', johansen_test.eig)
    print('Trace statistics:', johansen_test.lr1)
    print('Critical values (90%, 95%, 99%):', johansen_test.cvt[:, [0, 1, 2]])
    print('Max eigen statistics:', johansen_test.lr2)
    print('Critical values (90%, 95%, 99%):', johansen_test.cvm[:, [0, 1, 2]])

    mean = np.mean(residuals)
    std = np.std(residuals)

    z_scores = (residuals - mean) / std

    residuals = residuals.to_frame()
    stock1 = stock1.to_frame()
    stock2 = stock2.to_frame()

    df = pd.concat([stock1, stock2, residuals, z_scores], axis=1)
    df.columns = [f'{s1}_Close', f'{s2}_Close', 'Residual', 'Z-Score']

    print(df)
    df.plot(figsize=(8,8))
    
    # adding scatter plot
    plt.figure(figsize=(8,8))
    plt.scatter(df[f'{s1}_Close'], df[f'{s2}_Close'])
    plt.xlabel(f'{s1}_Close')
    plt.ylabel(f'{s2}_Close')
    plt.show()

    return df

for pair in pairs:
    df = analyse_pair(*pair)
