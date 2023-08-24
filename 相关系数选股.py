# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:48:54 2023

@author: wytheY
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_close_price(stock_code, number_of_days=60):
    df = pd.read_csv(f'E:/股市数据/a股日线/{stock_code}.csv', index_col = 0, encoding='gbk')
    return df.head(number_of_days)['收盘价'].values.tolist()

def create_heatmap(data, tick_labels):
    pccs = np.corrcoef(data)
    pd_data = pd.DataFrame(pccs, index=tick_labels, columns=tick_labels)
    mask = np.zeros_like(pd_data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 15)) 
    heatmap = sns.heatmap(pd_data,
                          mask=mask,
                          square=True,
                          linewidths=.5,
                          cmap='PuBuGn',
                          cbar_kws={'shrink': .4, 'ticks' : [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1,
                          annot=True,
                          annot_kws={"size": 12})

stock_codes = ['002277.SZ','600730.SH','300212.SZ','002279.SZ','600584.SH',
               '603986.SH','300001.SZ','603936.SH','600276.SH','600998.SH','000963.SZ',
               '600529.SH','002456.SZ','300146.SZ','002044.SZ','002241.SZ','300109.SZ',
               '600176.SH']
data = [get_close_price(code) for code in stock_codes]
create_heatmap(data, stock_codes)
