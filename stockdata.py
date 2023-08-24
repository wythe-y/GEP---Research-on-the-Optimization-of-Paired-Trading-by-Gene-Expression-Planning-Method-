import yfinance as yf
from talib import abstract
import talib
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt



def ta_features(df, indicators, period, start_year, end_year):
    df = df.rename(columns={
    "最高价": "high",
    "最低价": "low",
    "开盘价": "open",
    "收盘价": "close",
    "前收盘价": "adj close",
    "成交额(千元)": "volume",
    "交易时间": "date"
    })
    df['HO'] = df.high - df.open
    df['OL'] = df.open - df.low
    df['HCY'] = df.high - df.close.shift(1)
    df['CYL'] = df.close.shift(1) - df.low  
    df = fillTA(df, indicators, period)
    df = df.dropna()
    start_year_str = str(start_year)
    end_year_str = str(end_year + 1)
    start_date_str = start_year_str + '0101'
    end_date_str = end_year_str + '0101'
    df.date = pd.to_datetime(df.date, format='%Y%m%d')
    df.set_index('date', inplace=True)
    df = df.truncate(before=start_date_str, after=end_date_str)
    return df
    print(df)

def get_stock_data(sid, start, end, indicators, period):
    stock = '{}.TW'.format(sid)
    start_date = start
    end_date = end
    data = yf.download(stock, start_date, end_date)
    data.columns = ['open', 'high', 'low', 'close', 'adj close', 'volume']
    data['HO']=data.high-data.open
    data['OL']=data.open-data.low
    data['HCY']=data.high-data.close.shift(1)
    data['CYL']=data.close.shift(1)-data.low
    return fillTA(data, indicators, period)

def BIAS_indicator(data, p):
    ma = eval('abstract.SMA(data, timeperiod = {})'.format(p))
    bias = (data['close'] - ma)/ma*100
    return bias

def PSY_indicator(data, p):
    diff = data.diff()
    diff = np.append(0, diff)
    diff_dir = np.where(diff > 0, 1, 0)
    psy = np.zeros((len(data),))
    psy[:p] *= np.nan
    for i in range(p, len(data)):
        psy[i] = (diff_dir[i-p+1:i+1].sum()) / p
    return psy*100

     
def fillTA(df, indicators, period):
    dfhub = df
    modelName = 'RES'
    portfoilo = pd.read_csv('portfoilo/V1/{}_portfoilo.csv'.format(modelName)) #投資組合
    for i in range(len(portfoilo.columns)):
       #print(range(len(portfoilo.columns)))
        for j in portfoilo.iloc[:, i]:           
            if i == 0:
                this_1 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(j), encoding='gbk') 
                this_1 = this_1.dropna()
            elif i == 1:   
                this_2 = pd.read_csv(r'E:\股市数据\a股日线\{}.csv'.format(j), encoding='gbk') 
                this_2 = this_2.dropna()

    this_1.set_index('交易时间', inplace=True)
    this_2.set_index('交易时间', inplace=True)

    aligned_stock1, aligned_stock2 = this_1.align(this_2, join='inner')

    stock1 = aligned_stock1['收盘价']
    stock2 = aligned_stock2['收盘价']

    slope, intercept, _, _, _ = linregress(stock1, stock2)
    stock2_predicted = intercept + slope * stock1
    residuals = stock2 - stock2_predicted

    # 将Series转化为DataFrame
    residuals_df = residuals.to_frame()
    
        # 更改列名
    residuals_df.columns = ['close']
    
    # 重命名索引并重置为列
    residuals_df.index.name = 'date'
    residuals_df.reset_index(inplace=True)
    
    # 首先，我们假设你已经有了df 和 residuals_df
    
    # 创建df和residuals_df的副本
    df_copy = df.copy()
    res_copy = residuals_df.copy()
    
    # 在合并之前，我们要先确保'date'列是datetime类型
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    res_copy['date'] = pd.to_datetime(res_copy['date'])
    
    # 然后，我们将df_copy和res_copy进行合并，按照'date'列进行合并，并设置how='left'以保留df_copy中的所有日期
    resdf = pd.merge(df_copy, res_copy, on='date', how='left', suffixes=('_df', '_res'))
    
    # 使用residuals的数据替换df的'close'数据，如果residuals中没有对应的数据，就保留df的数据
    resdf['close'] = resdf['close_res'].where(resdf['close_res'].notna(), resdf['close_df'])
    
    # 删除多余的列
    resdf = resdf.drop(columns=['close_df', 'close_res'])
    
    # 为了保持resdf的'date'列的格式和内容与df一致，我们将从df复制'date'列到resdf
    resdf['date'] = df['date'].values
   #resdf.fillna(method='ffill', inplace=True)
    
    # 打印新的dataframe
   #print(resdf)
   
    per_min = period[0] # 最小天期
    per_max = period[1] + 1 # 最大天期
    d = dict()
    arr1 = []
    for ta in indicators:
        if ta == 'K' or ta == 'D':            
            for p in range(per_min, per_max):
                kd = eval('abstract.STOCH(df, fastk_period = {})'.format(p))
                K = 'K_{}'.format(p)
                D = 'D_{}'.format(p)
                kd.columns = [K, D]
                d[K] = kd[K]
                d[D] = kd[D]
        elif ta == 'BIAS':
            for p in range(per_min, per_max):
                bias = BIAS_indicator(df, p)
                col_name = '{}_{}'.format(ta, p)
                d[col_name] = bias
                 
        elif ta == 'AR':
            for p in range(per_min, per_max):
                col_name = '{}_{}'.format(ta, p)
                d[col_name] = talib.SUM(df.HO, timeperiod=p)/talib.SUM(df.OL, timeperiod=p)*100
        elif ta == 'BR':
            for p in range(per_min, per_max):
                col_name = '{}_{}'.format(ta, p)
                d[col_name] = talib.SUM(df.HCY, timeperiod=p)/talib.SUM(df.CYL, timeperiod=p)*100
        elif ta == 'Z-sorse':
            # 确保 'date' 列是日期类型
            # df['date'] = pd.to_datetime(df['date'])
            # residuals_df['date'] = pd.to_datetime(residuals_df['date'])
            
            # # 通过 'date' 列对齐 df 和 residuals_df
            # resdf = pd.merge(df, residuals_df, on='date', how='inner')
            epsilon = 1e-7
            for p in range(per_min, per_max):
                col_name = '{}_{}'.format(ta, p)
                MA =  resdf.close.rolling(window = p).mean()                
                STD =  resdf.close.rolling(window = p).std()+ epsilon
                resdf[col_name] = (resdf.close - MA) / STD
               #print(resdf[col_name],'XXXXXXXXXXXXXXXXXX656')
                
               #arr1.append(p)   
        else:
            for p in range(per_min, per_max):
                col_name = '{}_{}'.format(ta, p)
                d[col_name] = eval('abstract.{}(df, timeperiod = {})'.format(ta, p))
    print(resdf)     
    resdf = pd.DataFrame(resdf)             
    ta_df = pd.DataFrame(d)
    ta_df['date'] = df['date']
    # 创建列名列表
    #cols = ['date'] + ['Z-score_{}'.format(i) for i in range(1, 30)]
    
    # 选择所需列
    # resdf_selected = resdf[cols]
    # print(resdf_selected)
    # 合并
    intermediate_df = pd.merge(df, ta_df, on='date', how='inner')  # 'inner' 可以根据您的需要替换为 'outer', 'left', 或 'right'
    df = pd.merge(intermediate_df, resdf, on='date', how='inner')  # 'inner' 可以根据您的需要替换为 'outer', 'left', 或 'right'

    #df.fillna(0, inplace=True)
    df = df.drop(columns=["high_y","low_y","open_y","close_y","adj close_y","volume_y","成交量(手)_y",
"涨跌额_y","涨跌幅_y"])
    df = df.rename(columns={
    "high_x": "high",
    "low_x": "low",
    "open_x": "open",
    "close_x": "close",
    "adj close_x": "adj close",
    "volume_x": "volume",
    "成交量(手)_x": "成交量(手)",
    "涨跌额_x":"涨跌额",
    "涨跌幅_x":"涨跌幅" 
    })
    print(df)
    return df



# 创建一个时间序列
# date_rng = pd.date_range(start='1/1/2020', end='1/10/2020', freq='D')
# df = pd.DataFrame(date_rng, columns=['date'])
# s = ''
# df['data'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# df['data'] = pd.read_csv(r'E:/股市数据/a股日线/{}.csv'.format(s), index_col = 0, encoding='gbk')
# df['data'] = df['收盘价']
# # 设置日期为索引
# df.set_index('date', inplace=True)

# # 计算滑动窗口的平均值和标准差

# MA = df.close.rolling(window = p).mean()
# STD = df.close.rolling(window = p).std()

# # 计算Z-Score
# df[col_name] = (df.close - MA) / STD

# print(df)
