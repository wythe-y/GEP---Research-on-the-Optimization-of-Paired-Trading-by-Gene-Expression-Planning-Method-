# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:42:39 2023

@author: wytheY
"""

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
        B = 0.012016063811404057 # 假设B为2，可以根据实际情况调整,注意反转情况
    
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
            if trade_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    next_price_df2 = round(df2['收盘价'].iloc[i+1], 2) # 下期开盘价 df2
                    for _ in range(lot):
                        treasury_stock.append(next_price)
                        treasury_stock2.append(next_price_df2 * B) # 同时买入另一只股票的 B 倍张数
                        account -= df['open'].iloc[i+1] * 1000 + df2['收盘价'].iloc[i+1] * B * 1000
                        if df.index[i+1] in trade_detail:
                            trade_detail[df.index[i+1]][2] += 1
                        else:
                            trade_detail[df.index[i+1]] = [next_price, action_name, 1, 0]
                    state = 1 if len(treasury_stock) > 0 else 0
            if zsorse_signal == 1 and i != len(df)-1:
                check_ = check_money(df, i, account, lot)
                if check_:
                    action_name = 'buy'
                    next_price = round(df['open'].iloc[i+1], 2) # 下期開盤價
                    next_price_df2 = round(df2['收盘价'].iloc[i+1], 2) # 下期开盘价 df2
                    for _ in range(lot):
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
                action_name = 'stop profit stop loss'
                date_ = df.index[i+1]
                next_price = round(df['open'].iloc[i+1], 2)
                next_price_df2 = round(df2['收盘价'].iloc[i+1], 2) # 下期开盘价 df2
                while state == 1:       
                    p, r = get_stock_profit(treasury_stock[-1], next_price)
                    p1,r1 = get_stock_profit3(treasury_stock2[-1],next_price_df2,B)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    profit.append(p1)
                    rr.append(r1)
                    account += p1
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
                    p, r = get_stock_profit(treasury_stock[-1], current_price)
                    p1,r1 = get_stock_profit3(treasury_stock2[-1],current_price_df2,B)
                    profit.append(p)
                    rr.append(r)
                    account += p
                    profit.append(p1)
                    rr.append(r1)
                    account += p1
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
                    if df.index[i] in trade_detail:
                        newprofit = trade_detail[date_][3] + p
                        trade_detail[date_][2] += 1
                        trade_detail[date_][3] = newprofit
                    else:
                        trade_detail[date_] = [current_price, action_name, 1, p]
                    state = 1 if len(treasury_stock) > 0 else 0
            else: 
                if trade_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                   #check_2 = check_lot(treasury_stock2, lot)                    
                    if check_ :
                        date_ = df.index[i+1]
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        next_price_df2 = round(df2['收盘价'].iloc[i+1], 2)
                        for _ in range(lot):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            p, r = get_stock_profit(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                            min_price_index2 = treasury_stock2.index(min(treasury_stock2))
                            p1,r1 = get_stock_profit3(treasury_stock2.pop(min_price_index2),next_price_df2,B)
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
                if zsorse_signal == -1 and i != len(df)-1: # 賣出
                    check_ = check_lot(treasury_stock, lot)
                    if check_:
                        date_ = df.index[i+1]
                        next_price = df['open'].iloc[i+1] # 下期開盤價
                        next_price_df2 = round(df2['收盘价'].iloc[i+1], 2)
                        for _ in range(lot):
                            min_price_index = treasury_stock.index(min(treasury_stock))
                            min_price_index2 = treasury_stock2.index(min(treasury_stock2))                            
                            if MARK == 9:
                                p, r = get_short_stock_profit2(treasury_stock.pop(min_price_index), df['open'].iloc[i+1])
                                p1,r1 = get_stock_profit3(treasury_stock2.pop(min_price_index2),next_price_df2)
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