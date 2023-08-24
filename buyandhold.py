from gep2.stockdata import *
from gep2.tools import *
from gep2.fitness import *

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def FillStockData(portfoilo, start_year, end_year):
    new_dict = dict()
    start_year = '{}-01-01'.format(start_year)
    end_year = '{}-01-01'.format(end_year+1)
    for i in range(len(portfoilo)):
        for j in portfoilo.iloc[i]:
            this_ = pd.read_csv('gep2/stockdata/{}.csv'.format(j), index_col = 0)
            this_ = this_.dropna()
            new_dict[j] = this_.truncate(before=start_year, after=end_year)
    return new_dict

# 股票資料
modelName = 'LGBM'
portfoilo = pd.read_csv('gep2/portfoilo/V1/{}_portfoilo.csv'.format(modelName)) #投資組合

start_year, end_year = 2015, 2021 # 起始年 結束年
pf_dict = FillStockData(portfoilo, start_year, end_year) # 不同股票標的的盤後與指標資料

# 窗格設定
year = [y for y in range(start_year, end_year+1)]
#year = sorted(set(df.index.year))
windows = get_windows(year = year, start_year = start_year)
train_window_size = 3
test_window_size = 2
slide_window_size = test_window_size
 

period = 0 # 期數

總交易各期損益 = list()
總交易各期報酬率 = dict()

總交易損益= list()
總交易報酬率 = list()

def get_stock_profit(buy, sell): # 交易的損益(包含交易成本)
    buy = buy * 1000
    sell = sell * 1000
    commission_rate = 0.001425
    transfer_tax = 0.003
    cost = (buy + sell) * commission_rate + sell * transfer_tax + buy
    profit = sell - cost
    rr = round(profit/cost, 2)
    return profit, rr

money = [10000000]
money = [money[0]/len(portfoilo.columns) for _ in range(len(portfoilo.columns))]
for i in range(2, len(windows)-test_window_size-4, slide_window_size):
    print('period: ', period+1)
    s1, e1 = i, i + train_window_size
    s2, e2 = e1, e1 + test_window_size

    print('B&H time:', get_date(windows[s2]), '~', windows[e2])
    
    各期損益 = list()
    各期報酬率 = list()

    for stock in range(len(portfoilo.columns)): 
        df = pf_dict[portfoilo.iloc[period][stock]]
        df = df.truncate(before = get_date(windows[s2]), after = windows[e2])
        lot = 0
        buy = df.iloc[0].close * 1000
        while buy * lot * 1000 < money[stock]:
            lot += 1
        
        buy = df.iloc[0].close * 1000 * lot
        sell = df.iloc[-1].close * 1000 * lot
        commission_rate = 0.001425
        transfer_tax = 0.003
        cost = (buy + sell) * commission_rate + sell * transfer_tax + buy
        ret = sell - cost
        retRate = round(ret/cost, 2)
        各期損益.append(ret)
        各期報酬率.append(retRate)
    
    總交易各期損益.append(各期損益)
    總交易各期報酬率[get_date(windows[s2])] = [ ARR(各期報酬率),  # 每期平均報酬
                                              NetPorfit(各期損益), # 每期總淨利
                                              Odds(各期損益), # 勝率
                                              SharpRatio(len(df), 各期損益, 'daily'),
                                              MaxDrawdown(各期損益),
                                              MaxDrawdownRate(各期損益),
                                              RRR(各期損益, MaxDrawdown(各期損益)),
                                              ProfitFactor(各期損益),
                                              ProfitLossRatio(各期損益),
                                              Kelly(各期損益)]        
    
    總交易損益.extend(各期損益)
    總交易報酬率.extend(各期報酬率)
    
    period += 1

df = pd.DataFrame.from_dict(總交易各期報酬率, orient='index', columns=['平均報酬率',
                                                                      '總淨利',
                                                                      '勝率',
                                                                      '夏普值',
                                                                      'MDD',
                                                                      'MDD(Rate)',
                                                                      '風報比',
                                                                      '獲利因子',
                                                                      '盈虧比',
                                                                      '凱利值'])
df['total'] = df['平均報酬率'].cumsum()

df.total.plot() 
總淨利 = df['總淨利'].sum()
df.to_excel('{}買www入持有.xlsx'.format(modelName))