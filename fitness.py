from gep2.multigene import *
import numpy as np
import math
import copy

def fitness_function(individual, data, buy_threshold, sell_threshold):
    # Initialize portfolio and cash
    portfolio_high = 0  # Portfolio for high-priced stock
    portfolio_low = 0  # Portfolio for low-priced stock
    cash = 10000000  # Initial cash

    # Assume data is a list of tuples, each tuple contains (z_score, price_high, price_low)
    for z_score, price_high, price_low in data:
        # Decision tree
        if individual[0] == 'buy' and z_score > buy_threshold:
            # Short sell high-priced stock if cash is enough
            if cash > price_high:
                portfolio_high -= price_high
                cash -= price_high
                cost = price_high

            # Buy low-priced stock if cash is enough
            if cash > price_low:
                portfolio_low += price_low
                cash -= price_low
                cost += price_low

            # Compute current asset after buying
            current_asset = cash + portfolio_high * price_high + portfolio_low * price_low

            # Stop loss or take profit conditions
            if current_asset <= 0.9 * cost:  # Stop loss when total asset drops to 90% of cost
                print("Stop loss triggered, total asset: ", current_asset)
                return current_asset  # Return immediately

            if current_asset >= 1.4 * cost:  # Take profit when total asset raises to 140% of cost
                print("Take profit triggered, total asset: ", current_asset)
                return current_asset  # Return immediately

        elif individual[0] == 'sell' and z_score < sell_threshold:
            # Cover short position of high-priced stock (i.e., buy these stocks)
            if portfolio_high < 0:
                cash -= portfolio_high
                portfolio_high = 0

            # Sell low-priced stock if portfolio_low is not empty
            if portfolio_low > 0:
                cash += portfolio_low
                portfolio_low = 0

        # Do nothing if the action is 'do_nothing' or Z-score is in between the thresholds

    # Compute the total asset at the end
    total_asset = cash + portfolio_high * data[-1][1] + portfolio_low * data[-1][2]

    # The fitness is the total asset at the end (the more the better)
    fitness = total_asset

    return fitness

def tradeZsorse(chro, df, money, ta):
    account = money # 帳戶餘額
    state = 0 # 持有部位狀態
    profit = [] # 每次交易損益
    rr = [] # 每筆交易報酬率
    treasury_stock = [] # 庫存股存放
    trade_detail = dict() # 交易明細
    for i in range(len(df)):
        trade_signal, lot, stop_signal = get_action(chro, i, df, treasury_stock, ta)
        if state == 0:
            if trade_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    for _ in range(lot):
                        treasury_stock.append(next_price)
                        account -= df['open'].iloc[i+1] * 1000
                        if df.index[i+1] in trade_detail:
                            trade_detail[df.index[i+1]][2] += 1
                        else:
                            trade_detail[df.index[i+1]] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
        elif state == 1:
            if stop_signal and i != len(df)-1: # 停損停利
                action_name = 'stop profit stop loss'
                date_ = df.index[i+1]
                next_price = round(df['open'].iloc[i+1], 2)
                while state == 1:       
                    p, r = get_stock_profit(treasury_stock.pop(), next_price)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    if date_ in trade_detail:
                        trade_detail[date_][2] += 1
                    else:
                        trade_detail[date_] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
            else: 
                if trade_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                    if check_:
                        action_name = 'sell'
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                            profit.append(p)
                            rr.append(r)
                            account += p
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(df['open'].iloc[i+1]), action_name, 1]
                        state = 1 if len(treasury_stock)>0 else 0 
                elif trade_signal == 1 and i != len(df)-1:
                    check_ = check_money(df, i, account, 1)
                    if check_:
                        action_name = 'buy'
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            treasury_stock.append(next_price)
                            account -= df['open'].iloc[i+1] * 1000 
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(next_price), action_name, 1]
                        state = 1
    profit = [int(float(p)) for p in profit]
    return profit, rr, trade_detail

def fitness(chromosome, df, money, ta):
    profit, rr, trade_detail = trade2_one(chromosome, df, money, ta)
    info = {'total_rr': ROI(profit, money),
            'RRR': RRR(profit, MaxDrawdown(profit))}
    return info

def realTrade(chromosome, df, money, ta):
    profit, rr, trade_detail, money = trade_one(chromosome, df, money, ta)
    info = {'本金': money,
            '淨利': NetPorfit(profit),
            '勝率': Odds(profit),
            '交易次數': TradeCount(profit),
            '絕對報酬率': ROI(profit, money),
            '最大交易回落': MaxDrawdown(profit),
            '最大交易回落(%)': MaxDrawdownRate(profit),
            '風報比': RRR(profit, MaxDrawdown(profit)),
            '盈虧比': ProfitLossRatio(profit),
            '獲利因子': ProfitFactor(profit),
            '凱利值': Kelly(profit),
            '夏普值': SharpRatio(len(df), rr, period='daily', riskFree=0.018),
            'trade_detail': profit,
            'trade_date_detail': trade_detail}
    return info

def get_stock_profit(buy, sell): # 交易的損益(包含交易成本)
    buy = buy * 1000
    sell = sell * 1000
    commission_rate = 0.001425
    transfer_tax = 0.003
    cost = (buy + sell) * commission_rate + sell * transfer_tax + buy
    profit = sell - cost
    rr = round(profit/cost, 2)
    return profit, rr

def get_stock_profit3(buy, sell , B): # 交易的損益(包含交易成本)
    buy = buy * 1000
    sell = sell * 1000
    commission_rate = 0.001425
    transfer_tax = 0.003
    cost = (buy + sell) * commission_rate + sell * transfer_tax + buy
    profit = (sell - cost) * B
    rr = round(profit/(cost * B), 2)
    return profit, rr

def get_short_stock_profit2(buy,sell): # 做空交易的損益(包含交易成本)
    sell = sell * 1000
    buy = buy * 1000
    commission_rate = 0.001425
    transfer_tax = 0.003
    cost = (buy + sell) * commission_rate + buy * transfer_tax + sell
    profit = sell - cost
    rr = round(profit/cost, 2)
    return profit, rr

def get_short_stock_profit4(buy,sell,B): # 做空交易的損益(包含交易成本)
    sell = sell * 1000
    buy = buy * 1000
    commission_rate = 0.001425
    transfer_tax = 0.003
    cost = (buy + sell) * commission_rate + buy * transfer_tax + sell
    profit = (sell - cost)*B
    rr = round(profit/cost*B, 2)
    return profit, rr


def check_money(df, i, account, lot): # 檢查戶頭的錢是否足夠
    if int(df['open'].iloc[i].astype(float) * 1000) * int(lot) * 1.1 < account:
        return True
    else:
        return False

def check_lot(treasury_stock, lot): # 檢查庫存股張數是否夠賣
    if len(treasury_stock) > lot:
        return True
    else:
        return False


def trade2(chro, df, money, ta):
    print(chro,df ,money,ta)
    account = money # 帳戶餘額
    state = 0 # 持有部位狀態
    profit = [] # 每次交易損益
    rr = [] # 每筆交易報酬率
    treasury_stock = [] # 庫存股存放
    trade_detail = dict() # 交易明細
    for i in range(len(df)):
        trade_signal, lot, stop_signal = get_action(chro, i, df, treasury_stock, ta)
        if state == 0:
            if trade_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    for _ in range(lot):
                        treasury_stock.append(next_price)
                        account -= df['open'].iloc[i+1] * 1000
                        if df.index[i+1] in trade_detail:
                            trade_detail[df.index[i+1]][2] += 1
                        else:
                            trade_detail[df.index[i+1]] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
        elif state == 1:
            if stop_signal and i != len(df)-1: # 停損停利
                action_name = 'stop profit stop loss'
                date_ = df.index[i+1]
                next_price = round(df['open'].iloc[i+1], 2)
                while state == 1:       
                    p, r = get_stock_profit(treasury_stock.pop(), next_price)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    if date_ in trade_detail:
                        trade_detail[date_][2] += 1
                    else:
                        trade_detail[date_] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
            else: 
                if trade_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                    if check_:
                        action_name = 'sell'
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                            profit.append(p)
                            rr.append(r)
                            account += p
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(df['open'].iloc[i+1]), action_name, 1]
                        state = 1 if len(treasury_stock)>0 else 0 
                elif trade_signal == 1 and i != len(df)-1:
                    check_ = check_money(df, i, account, lot)
                    if check_:
                        action_name = 'buy'
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            treasury_stock.append(next_price)
                            account -= df['open'].iloc[i+1] * 1000
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(next_price), action_name, 1]
                        state = 1
    profit = [int(float(p)) for p in profit]
    return profit, rr, trade_detail


def trade2_one(chro, df, money, ta):
    account = money # 帳戶餘額
    state = 0 # 持有部位狀態
    state = 0
    profit = [] # 每次交易損益
    rr = [] # 每筆交易報酬率
    treasury_stock = [] # 庫存股存放
    trade_detail = dict() # 交易明細
   #print(df)
    for i in range(len(df)):
        trade_signal, lot, stop_signal,zsorse_signal,hedging = get_action(chro, i, df, treasury_stock, ta)
        if state == 0:
            if trade_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    for _ in range(lot):
                        treasury_stock.append(next_price)
                        account -= df['open'].iloc[i+1] * 1000
                        if df.index[i+1] in trade_detail:
                            trade_detail[df.index[i+1]][2] += 1
                        else:
                            trade_detail[df.index[i+1]] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
        elif state == 1:
            if stop_signal and i != len(df)-1: # 停損停利
                action_name = 'stop profit stop loss'
                date_ = df.index[i+1]
                next_price = round(df['open'].iloc[i+1], 2)
                while state == 1:       
                    p, r = get_stock_profit(treasury_stock.pop(), next_price)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    if date_ in trade_detail:
                        trade_detail[date_][2] += 1
                    else:
                        trade_detail[date_] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
            else: 
                if trade_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                    if check_:
                        action_name = 'sell'
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                            profit.append(p)
                            rr.append(r)
                            account += p
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(df['open'].iloc[i+1]), action_name, 1]
                        state = 1 if len(treasury_stock)>0 else 0 
    profit = [int(float(p)) for p in profit]
    return profit, rr, trade_detail

def rows_in_df(df1, df2):
    df1 = df1.to_frame()
    df2 = df2.to_frame()
    return set(df1.itertuples(index=False)).issubset(set(df2.itertuples(index=False)))
   #print(rows_in_df(df1, df2))

def trade_one(chro, df, money, ta):
   #print(df)
    modelName = 'RES'
    portfoilo = pd.read_csv('portfoilo/V1/{}_portfoilo.csv'.format(modelName)) #投資組合
    for i in range(len(portfoilo.columns)):
        for j in portfoilo.iloc[:, i]:           
            if i == 0:
                this_1 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(j), encoding='gbk') 
                this_1 = this_1.dropna()
            elif i == 1:   
                this_2 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(j), encoding='gbk') 
                this_2 = this_2.dropna()

    if rows_in_df(df['涨跌额'],this_1['涨跌额']) == True:
        df2 = this_2
        B = 0.012016063811404057 # 假设B为2，可以根据实际情况调整,注意反转情况
    else:
        df2 = this_1
        B = 0.5 # 假设B为2，可以根据实际情况调整,注意反转情况
    
    temp = money
    account = money # 帳戶餘額
    state = 0 # 持有部位狀態
    profit = [] # 每次交易損益
    rr = [] # 每筆交易報酬率
    treasury_stock = [] # 庫存股存放
    treasury_stock2 = [] # 另一只股票的庫存股存放
    trade_detail = dict() # 交易明細
    MARK = int()
    for i in range(len(df)):    
        trade_signal, lot, stop_signal, zsorse_signal, hedging = get_action(chro, i, df, treasury_stock, ta)
        if state == 0:
            # if trade_signal == 1 and i != len(df)-1:
            #     check_ = check_money(df, i, account, lot)
            #     if check_:
            #         action_name = 'buy'
            #         next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
            #         next_price_df2 = round(df2['收盘价'].iloc[i+1], 2) # 下期开盘价 df2
            #         for _ in range(lot):
            #             treasury_stock.append(next_price)
            #             treasury_stock2.append(next_price_df2 * B) # 同时买入另一只股票的 B 倍张数
            #             account -= df['open'].iloc[i+1] * 1000 + df2['收盘价'].iloc[i+1] * B * 1000
            #             if df.index[i+1] in trade_detail:
            #                 trade_detail[df.index[i+1]][2] += 1
            #             else:
            #                 trade_detail[df.index[i+1]] = [next_price, action_name, 1, 0]
            #         state = 1 if len(treasury_stock) > 0 else 0
          #print(zsorse_signal)
            if zsorse_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    next_price_df2 = round(df2['收盘价'].iloc[i+1], 2) # 下期开盘价 df2
                    for _ in range(6):
                        treasury_stock.append(next_price)
                        treasury_stock2.append(next_price_df2 * B) # 同时买入另一只股票的 B 倍张数
                        account -= df['open'].iloc[i+1] * 1000 + df2['收盘价'].iloc[i+1] * B * 1000
                        if next_price >= next_price_df2* B:
                            MARK = 9
                        else:
                            MARK = 8
                        if df.index[i+1] in trade_detail:
                            trade_detail[df.index[i+1]][2] += 1
                        else:
                            trade_detail[df.index[i+1]] = [next_price, action_name, 1, 0]
                    state = 1 if len(treasury_stock) > 0 else 0
        elif state == 1:
            if stop_signal and i != len(df)-1: # 停損停利
               #print(stop_signal)
                action_name = 'stop profit stop loss'
                date_ = df.index[i+1]
                next_price = round(df['open'].iloc[i+1], 2)
                next_price_df2 = round(df2['收盘价'].iloc[i+1], 2) # 下期开盘价 df2
                while state == 1:       
                    # p, r = get_stock_profit(treasury_stock[-1], next_price)
                    # p1,r1 = get_stock_profit3(treasury_stock2[-1],next_price_df2,B)
                    # profit.append(p)
                    # rr.append(r)
                    # account += p
                    # profit.append(p1)
                    # rr.append(r1)
                    # account += p1
                    if MARK == 9:
                        p, r = get_short_stock_profit2(treasury_stock.pop(), next_price)
                        p1,r1 = get_stock_profit3(treasury_stock2.pop(),next_price_df2,B)
                        profit.append(p)
                        rr.append(r)
                        account += p
                        profit.append(p1)
                        rr.append(r1)
                        account += p1
                    else:
                        p, r = get_stock_profit(treasury_stock.pop(), next_price)
                        p1,r1 = get_short_stock_profit4(treasury_stock2.pop(),next_price_df2,B)
                        profit.append(p)
                        rr.append(r)
                        account += p
                        profit.append(p1)
                        rr.append(r1)
                        account += p1
                    if date_ in trade_detail:
                        newprofit = trade_detail[date_][3] + p
                        trade_detail[date_][2] += 1
                        trade_detail[date_][3] = newprofit
                    else:
                        trade_detail[date_] = [next_price, action_name, 1, p]
                    state = 1 if len(treasury_stock) > 0 else 0
            
            elif i == len(df)-1: # 時間到全部出場
                action_name = 'time up sell all'
                date_ = df.index[i]
                current_price = round(df['close'].iloc[i], 2)
                current_price_df2 = round(df2['收盘价'].iloc[i], 2)
                while state == 1:
                    # p, r = get_stock_profit(treasury_stock[-1], current_price)
                    # p1,r1 = get_stock_profit3(treasury_stock2[-1],current_price_df2,B)
                    # profit.append(p)
                    # rr.append(r)
                    # account += p
                    # profit.append(p1)
                    # rr.append(r1)
                    # account += p1
                    if MARK == 9:
                        p, r = get_short_stock_profit2(treasury_stock.pop(), current_price)
                        p1,r1 = get_stock_profit3(treasury_stock2.pop(),current_price_df2,B)
                        profit.append(p)
                        rr.append(r)
                        account += p
                        profit.append(p1)
                        rr.append(r1)
                        account += p1
                    else:
                        p, r = get_stock_profit(treasury_stock.pop(), current_price)
                        p1,r1 = get_short_stock_profit4(treasury_stock2.pop(),current_price_df2,B)
                        profit.append(p)
                        rr.append(r)
                        account += p
                        profit.append(p1)
                        rr.append(r1)
                        account += p1
                    if df.index[i] in trade_detail:
                        newprofit = trade_detail[date_][3] + p
                        trade_detail[date_][2] += 1
                        trade_detail[date_][3] = newprofit
                    else:
                        trade_detail[date_] = [current_price, action_name, 1, p]
                    state = 1 if len(treasury_stock) > 0 else 0
            else: 
                # if trade_signal == -1 and i != len(df)-1: # 賣出
                #     check_ = check_lot(treasury_stock, lot)
                #     #check_2 = check_lot(treasury_stock2, lot)                    
                #     if check_ :
                #         date_ = df.index[i+1]
                #         next_price = df['open'].iloc[i+1] # 下期開盤價
                #         next_price_df2 = round(df2['收盘价'].iloc[i+1], 2)
                #         for _ in range(lot):
                #             min_price_index = treasury_stock.index(min(treasury_stock))
                #             p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                #             min_price_index2 = treasury_stock2.index(min(treasury_stock2))
                #             p1,r1 = get_stock_profit3(treasury_stock2.pop(min_price_index2),next_price_df2,B)
                #             profit.append(p)
                #             rr.append(r)
                #             account += p
                #             profit.append(p1)
                #             rr.append(r1)
                #             account += p1
                #             if df.index[i+1] in trade_detail:
                #                 newprofit = trade_detail[date_][3] + p
                #                 trade_detail[date_][2] += 1
                #                 trade_detail[date_][3] = newprofit
                #             else:
                #                 trade_detail[df.index[i+1]] = [int(next_price), 'sell', 1, p]
                #         state = 1 if len(treasury_stock)>0 else 0 
                if zsorse_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                    if check_:
                        date_ = df.index[i+1]
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        next_price_df2 = round(df2['收盘价'].iloc[i+1], 2)
                        for _ in range(6):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            min_price_index2 = treasury_stock2.index(min(treasury_stock2))                            
                            if MARK == 9:
                                p, r = get_short_stock_profit2(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                                p1,r1 = get_stock_profit3(treasury_stock2.pop(min_price_index2),next_price_df2,B)
                                profit.append(p)
                                rr.append(r)
                                account += p
                                profit.append(p1)
                                rr.append(r1)
                                account += p1
                            else:
                                p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                                p1,r1 = get_short_stock_profit4(treasury_stock2.pop(min_price_index2),next_price_df2,B)
                                profit.append(p)
                                rr.append(r)
                                account += p
                                profit.append(p1)
                                rr.append(r1)
                                account += p1
                            if df.index[i+1] in trade_detail:
                                newprofit = trade_detail[date_][3] + p
                                trade_detail[date_][2] += 1
                                trade_detail[date_][3] = newprofit
                            else:
                                trade_detail[df.index[i+1]] = [int(next_price), 'sell', 1, p]
                        state = 1 if len(treasury_stock)>0 else 0 
    profit = [int(float(p)) for p in profit]
    return profit, rr, trade_detail, temp





def trade(chro, df, money, ta):
    account = money # 帳戶餘額
    state = 0 # 持有部位狀態
    profit = [] # 每次交易損益
    rr = [] # 每筆交易報酬率
    treasury_stock = [] # 庫存股存放
    trade_detail = dict() # 交易明細
    for i in range(len(df)):
        trade_signal, lot, stop_signal = get_action(chro, i, df, treasury_stock, ta)
        if state == 0:
            if trade_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    for _ in range(lot):
                        treasury_stock.append(next_price)
                        account -= df['open'].iloc[i+1] * 1000
                        if df.index[i+1] in trade_detail:
                            trade_detail[df.index[i+1]][2] += 1
                        else:
                            trade_detail[df.index[i+1]] = [next_price, action_name, 1]
                    state = 1 if len(treasury_stock) > 0 else 0
        elif state == 1:
            if stop_signal and i != len(df)-1: # 停損停利
                action_name = 'stop profit stop loss'
                date_ = df.index[i+1]
                next_price = round(df['open'].iloc[i+1], 2)
                while state == 1:       
                    p, r = get_stock_profit(treasury_stock.pop(), next_price)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    if date_ in trade_detail:
                        trade_detail[date_][2] += 1
                    else:
                        trade_detail[date_] = [next_price, action_name, 1, sum(profit)]
                    state = 1 if len(treasury_stock) > 0 else 0
            elif i == len(df)-1: # 時間到全部出場
                action_name = 'time up sell all'
                date_ = df.index[i]
                current_price = round(df['close'].iloc[i], 2)
                while state == 1:
                    p, r = get_stock_profit(treasury_stock.pop(), current_price)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    if df.index[i] in trade_detail:
                        trade_detail[date_][2] += 1
                    else:
                        trade_detail[date_] = [current_price, action_name, 1, sum(profit)]
                    state = 1 if len(treasury_stock) > 0 else 0
            else: 
                if trade_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                    if check_:
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                            profit.append(p)
                            rr.append(r)
                            account += p
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(df['open'].iloc[i+1]), 'sell', 1, sum(profit)]
                        state = 1 if len(treasury_stock)>0 else 0 
                elif trade_signal == 1 and i != len(df)-1:
                    check_ = check_money(df, i, account, lot)
                    if check_:
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        for _ in range(lot):
                            treasury_stock.append(next_price)
                            account -= df['open'].iloc[i+1] * 1000
                            if df.index[i+1] in trade_detail:
                                trade_detail[df.index[i+1]][2] += 1
                            else:
                                trade_detail[df.index[i+1]] = [int(next_price), 'buy', 1]
                        state = 1
    
    profit = [int(float(p)) for p in profit]
    return profit, rr, trade_detail
        
def route_rule(name, currentVal, minVal, maxVal, ta): # 規則走法
    route = -1
    minval, maxval = minVal, maxVal
    if minVal < maxVal:
        minval, maxval = maxVal, minVal
    total  = abs(ta[name][0]) + abs(ta[name][1])
    minval = total * minval + ta[name][0]
    maxval = total * maxval + ta[name][0]
    if currentVal <= minval:
        route = 0
    elif currentVal > minval and currentVal < maxval:
        route = 1
    elif currentVal >= maxval:
        route = 2
    return route

def get_signal(gene, data, date, ta): # 基因1的交易訊號
    current_node = gene[0] # first step is from root node
    i = 0
    while current_node.arity > 0:
        minVal, maxVal = current_node.minval, current_node.maxval # 當前指標值最小 當前指標值最大
        indicator, period = current_node.name, current_node.period # 技術指標名稱 技術指標天期
        col = '{}_{}'.format(indicator, period)
        val = data[col].iloc[date]
        route = route_rule(indicator, val, minVal, maxVal, ta)
        current_node = current_node.child[route]
        i += 1
    signal, lot = current_node.name, current_node.lot
    return signal, lot

def get_signalzsorse(gene, data, date, ta): # 基因2的交易訊號
    current_node = gene[0] # first step is from root node
    i = 0
    while current_node.arity > 0:
        minVal, maxVal = current_node.minval, current_node.maxval # 當前指標值最小 當前指標值最大
        indicator, period = current_node.name, current_node.period # 技術指標名稱 技術指標天期
        col = '{}_{}'.format(indicator, period)
        val = data[col].iloc[date]        
        route = route_rule(indicator, val, minVal, maxVal, ta)
        current_node = current_node.child[route]
        i += 1
    signal, lot = current_node.name, current_node.lot
    return signal, lot

def stop_profit_loss(gene, treasury_stock, date, data): # 停損以及停利
    signal = False
    if len(treasury_stock) > 0:
        avg_close = np.mean(treasury_stock)
        stop_loss, stop_profit = gene[0].risk, gene[1].risk
        current_close = data['close'].iloc[date]
        profit, rr = get_stock_profit(avg_close, current_close) 
        if rr*100 >  stop_profit: # stop profit
            signal = True
        elif rr*100 > stop_loss: # stop loss
            signal = True
    return signal
            
def get_action(chro, date, data, treasury_stock, ta):
    trade_signal = 0
    stop_signal = 0
    lot = 0
    hedging = 0
   #print(date,'xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    for gene in chro:
        if isinstance(gene, TradeGene):
            trade_signal, lot = get_signal(gene.genome, data, date, ta) # 交易訊號     
        elif isinstance(gene, RiskGene):
            stop_signal = stop_profit_loss(gene.genome, treasury_stock, date, data)
        elif isinstance(gene, ZsorseGene):
            zsorse_signal, hedging = get_signalzsorse(gene.genome, data, date, ta) # 交易訊號   
   #print(hedging,'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    return trade_signal, lot, stop_signal,zsorse_signal,hedging


def NetPorfit(profit): # 淨利
    if len(profit) > 0:
        profit = [p for p in profit if p!=0]
        return round(sum(profit), 0)
    else:
        return 0

def TradeCount(profit): # 交易次數
    if len(profit) > 0:
        profit = [p for p in profit if p != 0]
        return len(profit)
    else:
        return 0

def Odds(profit): # 勝率
    if len(profit) > 0:
        profit = [p for p in profit if p != 0]
        winTimes = len([p for p in profit if p > 0])
        loseTimes = len([p for p in profit if p < 0])
        odds = winTimes/(winTimes+loseTimes)
        return round(odds, 3)
    else:
        return 0


def ARR(returns): # 平均報酬
    if len(returns) > 0:
        returns = [ret for ret in returns if ret != 0]
        avgret = sum(returns)/len(returns) if len(returns) != 0 else 0
        return avgret
    else:
        return 0
    

def ProfitFactor(profit): # 獲利因子
    if len(profit) > 0:
        P = sum([i for i in profit if i > 0]) # total profit
        L = abs(sum([i for i in profit if i < 0])) # total loss
        pf = P/L if L != 0 else P/(L+1)
        return round(pf, 3)
    else:
        return 0


def IRR(tradelen, profit, money, period='daily'): # 年化報酬
    if period == 'daily':
        year = tradelen/252
    elif period == 'year':
        year = tradelen    
    if len(profit) > 0:
        roi = ROI(profit, money)
        irr = pow((1 + roi), 1/year)-1
        return irr
    else:
        return 0

def SharpRatio(tradelen, returns, period='daily', riskFree=0.018): # 夏普值
    if period == 'daily':
        period = 252/tradelen
    elif period == 'season':
        period = 2
    if len(returns) > 0:
        profit = ARR(returns) # 平均報酬
        profit = profit * period - riskFree # 轉年化並減去無風險利率
        std = np.std(returns) * math.sqrt(period) # 
        sp = profit/std if std != 0 else 0
        return sp
    else:
        return 0 

def ROI(profit, money): # 絕對報酬率
    if len(profit) > 0:
        return sum(profit)/money  
    else:
        return 0

def ProfitLossRatio(profit): # 盈虧比
    if len(profit) > 0:
        profit = [p for p in profit if p != 0]
        total_profit = [i for i in profit if i > 0] # total profit
        #total_loss = [i for i in profit if i < 0] # total loss
        total_loss = [abs(i) for i in profit if i < 0] # total loss
        avg_profit = sum(total_profit)/len(total_profit) if len(total_profit) != 0 else 0 # avg profit
        avg_loss = sum(total_loss)/len(total_loss) if len(total_loss) != 0 else 0 # avg loss
        #avg_loss = abs(sum(total_loss)/len(total_loss)) if len(total_loss) != 0 else 0 # avg loss
        plr = avg_profit/avg_loss if avg_loss != 0 else avg_profit/(avg_loss+1)
        return round(plr, 3)
    else:
        return 0

def Kelly(profit): # 凱莉值
    if len(profit) > 0:
        winRate = Odds(profit) # 勝率
        lossRate = 1-winRate # 敗率
        plr = ProfitLossRatio(profit) # 盈虧比
        kv = winRate - (lossRate/plr) if plr != 0 else 0
        return kv
    else:
        return 0

def MaxDrawdown(profit): # 最大交易回落
    if len(profit) < 1:
        return 0
    profit = np.cumsum(profit)
    i = np.argmax((np.maximum.accumulate(profit) - profit)/np.maximum.accumulate(profit))
    if i == 0:
        return 0
    k = np.argmax(profit[:i])
    drawdown_max = profit[k] - profit[i]
    return drawdown_max

def MaxDrawdownRate(profit): # 最大交易回落
    #print()
    if len(profit) < 1:
        return 0
    profit = np.cumsum(profit)
    i = np.argmax((np.maximum.accumulate(profit) - profit)/np.maximum.accumulate(profit))
    if i == 0:
        return 0
    k = np.argmax(profit[:i])
    drawdown_rate = (profit[k] - profit[i])/(profit[k])
    return drawdown_rate

def RRR(profit, mdd): # 風險報酬比
    if len(profit) > 0:
        totalprofit = NetPorfit(profit)
        RRR = totalprofit/mdd if mdd != 0 else totalprofit/(mdd+1)
        return RRR    
    else:
        return 0
        
        
        