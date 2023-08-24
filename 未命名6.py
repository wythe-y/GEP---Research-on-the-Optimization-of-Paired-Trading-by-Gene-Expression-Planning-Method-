# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 01:02:00 2023

@author: wytheY
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller

def analyse_pair(s1, s2):
    stock1 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(s1), encoding='gbk')
    stock2 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(s2), encoding='gbk')

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

    residuals = residuals.to_frame()
    stock1 = stock1.to_frame()
    stock2 = stock2.to_frame()

    df = pd.concat([stock1, stock2, residuals, z_scores], axis=1)
    df.columns = [f'{s1}_Close', f'{s2}_Close', 'Residual', 'Z-Score']

    df = df.loc['2015-01-01':'2020-12-31']
    
    df['PortfolioValue'] = 0.0  # Adding a column for portfolio value

    cash = 100000000  
    shares = 0  
    position = 0  
    trade_log = []
    shares_per_trade = 200  
    stop_loss = None

    for date, row in df.iterrows():
        z_score = row['Z-Score']
        current_value = cash + abs(shares) * row[f'{s1}_Close']
        df.loc[date, 'PortfolioValue'] = current_value  # Record portfolio value for each day

        if stop_loss is not None and current_value < stop_loss:
            if position == 1:
                cash += shares * row[f'{s1}_Close']
            elif position == -1:
                cash += shares * row[f'{s1}_Close']
            shares = 0
            position = 0
            stop_loss = None
            trade_log.append((date, cash))
            continue

        if z_score > 1.5 and position <= 0:
            if position == -1:
                cash += shares * row[f'{s1}_Close']
                shares = 0
            shares -= shares_per_trade
            cash -= shares_per_trade * row[f'{s1}_Close']
            position = 1
            stop_loss = current_value * 0.9
            trade_log.append((date, cash))

        elif z_score < 1.5 and position >= 0:
            if position == 1:
                cash += shares * row[f'{s1}_Close']
                shares = 0
            shares += shares_per_trade
            cash -= shares_per_trade * row[f'{s1}_Close']
            position = -1
            stop_loss = current_value * 0.9
            trade_log.append((date, cash))

        elif z_score == 0 and position != 0:
            if position == 1:
                cash += shares * row[f'{s1}_Close']
            elif position == -1:
                cash += shares * row[f'{s1}_Close']
            shares = 0
            position = 0
            stop_loss = None
            trade_log.append((date, cash))

    final_value = cash + abs(shares) * df.iloc[-1][f'{s1}_Close']
    total_return = final_value - 100000000

    df['Return'] = df['PortfolioValue'].pct_change()

    start_date = pd.to_datetime(df.index[0])
    end_date = pd.to_datetime(df.index[-1])
    years = (end_date - start_date).days / 365.25 + 1e-6

    if years > 0:
        annual_return = ((final_value / 100000000) ** (1 / years)) - 1
    else:
        annual_return = np.nan

    returns = df['Return'].dropna()
    annual_sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    num_trades = len(trade_log)

    roll_max = df['PortfolioValue'].cummax()
    daily_drawdown = df['PortfolioValue'] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.min()

    profits = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    profit_factor = profits / abs(losses)

    wins = returns[returns > 0].count()
    losses = returns[returns < 0].count()
    win_loss_ratio = wins / losses if losses != 0 else np.nan

    win_rate = wins / (wins + losses)
    loss_rate = losses / (wins + losses)
    p = win_rate
    q = 1 - p
    kelly_criterion = (p - q) / (wins - losses)

    metrics = {
        'Annual Return': annual_return,
        'Annual Sharpe Ratio': annual_sharpe_ratio,
        'Total Return': total_return,
        'Number of Trades': num_trades,
        'Max Daily Drawdown': max_daily_drawdown,
        'Profit Factor': profit_factor,
        'Win/Loss Ratio': win_loss_ratio,
        'Kelly Criterion': kelly_criterion
    }
    
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.index.name = 'Metric'

    return df, trade_log, metrics_df

s1 = '600998.SH'
s2 = '300146.SZ'
df, trade_log, metrics_df = analyse_pair(s1, s2)

print(df)
print('\nTrade Log:')
for date, cash in trade_log:
    print(f'{date}: {cash}')

print('\nMetrics:')
metrics_df = metrics_df.applymap('{:,.4f}'.format)
print(metrics_df)
