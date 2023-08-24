from gep2.node import *
from gep2.primitiveset import *
from gep2.multigene import *
from gep2.chromosome import *
from gep2.population import *
from gep2.stockdata import *
from gep2.tools import *
from gep2.fitness import *
import pandas as pd
import os
import sys
import time
import copy
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller


def fill_data(portfoilo, start_year, end_year):
    new_dict = dict()
    new_dictZ = dict()
    for i in range(len(portfoilo)):
        for j in portfoilo.iloc[i]:
            this_ = pd.read_csv(r'E:/股市数据/a股日线/{}.csv'.format(j), index_col = 0,encoding=('gbk'))
            this_ = this_.dropna()
            new_dict[j] = ta_features(this_, indicators, indicators_period, start_year, end_year)
            new_dictZ[j] = ta_features(this_, zsorse, zsorse_period, start_year, end_year)
    return new_dict,new_dictZ

def createFolder(model, size, iter_):
    file_route = 'record/{}/'.format(model)
    folder_count = 0 
    for file_num in os.listdir(file_route): # 計算路徑下的檔案
        if os.path.isdir(os.path.join(file_route, file_num)): # 如果是資料夾的話
            folder_count += 1 # 更新檔名編號
    fileName = file_route + str(folder_count + 1) + '_size{0}_iter{1}'.format(size, iter_) # 檔名
    os.mkdir(fileName)
    return fileName

def saveTradeDetail(route, stockID, detail, performance):
    fname = '{0}/{1}.xlsx'.format(route, stockID)

    with pd.ExcelWriter(fname, engine='xlsxwriter') as writer:   
        df_detail = pd.DataFrame.from_dict(detail, orient="index", columns = ['price', 'action', 'lot', 'profit'])
        df_performance = pd.DataFrame.from_dict(performance, orient="index")
        df_detail.to_excel(writer, sheet_name='detail')
        writer.sheets['detail'].set_column(0, 0, 20)
        df_performance.to_excel(writer, sheet_name='performance')
        writer.sheets['performance'].set_column(0, 0, 25)
   
def saveTotalTrade(route, filename, detail, performance, trade_detail):  
     fname = '{0}/{1}.xlsx'.format(route, filename)
     with pd.ExcelWriter(fname, engine='xlsxwriter') as writer:   
         df_total = pd.DataFrame.from_dict(detail, orient="index", columns = ['當期平均報酬率', '當期損益', '當期勝率',
                                                                              '當期交易次數', '當期年化報酬率', '當期年化夏普值',                                                                              
                                                                              '當期 MDD', '當期 MDD(%)', '風報比', '獲利因子',
                                                                              '盈虧比', '凱利值'])
         df_performance = pd.DataFrame.from_dict(performance, orient="index")
         df_trade_detail = pd.DataFrame.from_dict(trade_detail, orient="index")
         df_total.to_excel(writer, sheet_name='各期績效')
         df_performance.to_excel(writer, sheet_name='總績效')
         df_trade_detail.to_excel(writer, sheet_name='總交易紀錄')
     
def PeriodProfit(ary):
    alldates = sorted(set(np.concatenate([list(d.keys()) for d in ary]))) # 找出當期有交易的時間並排序
    sumprofit = list()
    for date in alldates:
        current_info = [d.get(date,0) for d in ary] # 當天買賣情況
        current_date_profit = sum([profit[3] for profit in current_info if isinstance(profit, list) and profit[3] != 0]) # 如果當天有買賣且有損益的話
        if current_date_profit != 0:
            sumprofit.append(current_date_profit)
    return sumprofit

def PeriodProfitDetail(ary):
    alldates = sorted(set(np.concatenate([list(d.keys()) for d in ary]))) # 找出當期有交易的時間並排序
    sumProfitDetail = dict()
    for date in alldates:
        current_info = [d.get(date,0) for d in ary] # 當天買賣情況
        current_date_profit = sum([profit[3] for profit in current_info if isinstance(profit, list) and profit[3] != 0]) # 如果當天有買賣且有損益的話
        if current_date_profit != 0:
            sumProfitDetail[date] = current_date_profit
    return sumProfitDetail



    
    
if __name__ == '__main__':
    
    
    indicators = ['K', 'D', 'RSI', 'BIAS', 'WILLR'] # 指標種類
    indicators_period = [5, 20] # 指標天期
    
    
    zsorse = ['Z-sorse'] # z-sorse
    zsorse_period = [10,30] #zcore天期
    
    # 股票資料包含技術指標
    modelName = 'RES'
    portfoilo = pd.read_csv('portfoilo/V1/{}_portfoilo.csv'.format(modelName)) #投資組合
    #portfoilo = [6414, 6409, 4426, 8446, 2228] # 投資組合
    start_year, end_year = 2015, 2021 # 起始年 結束年
    pf_dict,pf_dictz = fill_data(portfoilo, start_year, end_year) # 不同股票標的的盤後與指標資料

    # 窗格設定
    year = [y for y in range(start_year, end_year+1)]
    #year = sorted(set(df.index.year))
    windows = get_windows(year = year, start_year = start_year)
    train_window_size = 3
    test_window_size = 2
    slide_window_size = test_window_size
        
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
    
    # 假设你有一个包含日期的Series，你可以将其添加为新的列
    # 假设dates是一个包含交易日期的Series
    residuals_df.index.name = 'date'
    residuals_df.reset_index(inplace=True)
    residuals_df['date'] = pd.to_datetime(residuals_df['date'], format='%Y%m%d')

   #print(residuals_df)
    
    # 基因設定
    # 基因1, 2, 3
    lot = [1, 1] # 買賣張數
    trade_gene = Gene(name='TradeGene', pset=set_trade_gene(indicators, indicators_period, lot), head=2)
    dc_period = GeneDc(dcName='Period', dcLen=trade_gene.head, rncLen=trade_gene.head*2, threshold=indicators_period)
    dc_threshold_min = GeneDc(dcName='Threshold_min', dcLen=trade_gene.head, rncLen=trade_gene.head*2, threshold=[0, 1])
    dc_threshold_max = GeneDc(dcName='Threshold_max', dcLen=trade_gene.head, rncLen=trade_gene.head*2, threshold=[0, 1])
    # 由於頭部長度會縮減因此買賣張數dc為 尾部長度 + 頭部長度 - 1
    dc_Lot = GeneDc(dcName='Lot', dcLen=trade_gene.tail + trade_gene.head - 1, rncLen=trade_gene.tail*2, threshold=lot)
    trade_gene.add_dc_rnc([dc_period, dc_threshold_min, dc_threshold_max, dc_Lot])
    
    threshold = [-3,3]
    zsorse_gene = Gene(name='Z-sorseGene', pset=set_zsorse_gene(zsorse,zsorse_period), head=1)
    dc_threshold_min = GeneDc(dcName='threshold_min', dcLen=zsorse_gene.head, rncLen=zsorse_gene.head*2, threshold=threshold)
    dc_threshold_max = GeneDc(dcName='threshold_max', dcLen=zsorse_gene.head, rncLen=zsorse_gene.head*2, threshold=threshold)
    dc_zsorseperiod = GeneDc(dcName='zsorse_period', dcLen=zsorse_gene.head, rncLen=zsorse_gene.head*2, threshold=zsorse_period)
    zsorse_gene.add_dc_rnc([dc_zsorseperiod,dc_threshold_min, dc_threshold_max])
   #print(zsorse_gene)
    # 基因4
    risk_threshold = [1, 40] # 停損停利閥值 %
    risk_gene = Gene(name='RiskGene', pset=set_risk_gene(risk_threshold), head=0)
    dc_risk = GeneDc(dcName='Risk', dcLen=risk_gene.tail, rncLen=risk_gene.tail*2, threshold=risk_threshold)
    risk_gene.add_dc_rnc([dc_risk])
    # 染色體 
    chro = Chromosome([trade_gene, zsorse_gene, risk_gene])
    popSize = 50
    popIter = 50
    population = Population(chro=chro.genes, popSize=popSize, popIter=popIter)
    allevaluateDict = list()
    for run in range(5):
        # 初始資金
        moneyFortrain, moneyFortest = [10000000], [10000000]
        train_profit_record = dict() # 紀錄訓練期資金
        test_profit_record = dict() # 紀錄測試期資金
        if len(portfoilo.columns) > 0:
            moneyFortrain = [moneyFortrain[0]/len(portfoilo.columns) for _ in range(len(portfoilo.columns))]
            moneyFortest = [moneyFortest[0]/len(portfoilo.columns) for _ in range(len(portfoilo.columns))]
        
        
        
        filename = createFolder(modelName, popSize, popIter) # 創建資料夾
        subfilename = filename + '/detail_trade_history'
        os.mkdir(subfilename)
        os.mkdir(filename + '/total_trade_history')
        start = time.time() # 紀錄時間
        period = 0 # 期數
        new = 0 # 文件
        trainodds = list()
        testodds = list()
        
        總交易損益_訓練期 = list()
        總交易損益_測試期 = list()
        
        總交易報酬_訓練期 = list()
        總交易報酬_測試期 = list()
        
        總交易各期損益_訓練期 = list()  
        總交易各期損益_測試期 = list()
        
        總交易各期損益最大交易回落訓練期 = list()  
        總交易各期損益最大交易回落測試期 = list()  
        
        for i in range(2, len(windows)-test_window_size-4, slide_window_size):
            print('period: ', period+1)
            s1, e1 = i, i + train_window_size
            s2, e2 = e1, e1 + test_window_size
            
            print('train time:', windows[s1], '~', windows[e1])
            print('test time:', get_date(windows[s2]), '~', windows[e2])
            
            period_folder_name = subfilename + '/period_' + str(period+1) + '/new_' + str(new+1)
            os.makedirs(period_folder_name)
            
            train_period_folder_name = period_folder_name + '/train'
            os.mkdir(train_period_folder_name)
            
            test_period_folder_name = period_folder_name + '/test'
            os.mkdir(test_period_folder_name)
            
            evolutionfolder = period_folder_name + '/evolution'
            os.mkdir(evolutionfolder)
    
            genefolder = period_folder_name + '/gene'
            os.mkdir(genefolder)
    
            train_profit = list()
            test_profit = list()
            train_rr = list()
            test_rr = list()
            train_profit_ary = list()
            test_profit_ary = list()
            
                
            各期損益_訓練期 = list()
            各期損益_測試期 = list()
            
            各期報酬率_訓練期 = list()
            各期報酬率_測試期 = list()
            
            各期損益MDD_訓練期 = list()
            各期損益MDD_測試期 = list()
            
            for j in range(len(portfoilo.columns)):    
                # print(type(pf_dict))
                # print(pf_dict)
                # print(portfoilo.iloc[period][j])
                #
                df = pf_dict[portfoilo.iloc[period][j]]
                df2 = pf_dictz[portfoilo.iloc[period][j]]                
                df = df.combine_first(df2)

                

                print(df)
                #windows = pd.to_datetime(windows, format='%Y%m%d')
                start_date = pd.to_datetime(windows, format='%Y%m%d')[s1].strftime('%Y-%m-%d')
                end_date = pd.to_datetime(get_date(windows[e1]), format='%Y%m%d').strftime('%Y-%m-%d')
                train_df = df.truncate(before = start_date, after = end_date)#训练资料
                test_df = df.truncate(before = get_date(windows[s2]), after = windows[e2]) # 測試資料                
                print(test_df,train_df)
                ta1 = find_indicator_MinMax(train_df, indicators, indicators_period) # 指標極大極小表
                ta2 = find_indicator_MinMax(train_df, zsorse, zsorse_period) # Z-sorse指標的極大極小表
                ta = ta1.copy()  # 创建 ta1 的副本
                ta.update(ta2)  # 将 ta2 的数据更新到 ta 中
                #print(ta)
                # df = pf_dict[portfoilo.iloc[period][j]]
                # train_df = df.truncate(before = windows[s1], after = get_date(windows[e1])) # 訓練資料
                # test_df = df.truncate(before = get_date(windows[s2]), after = windows[e2]) # 測試資料
                # ta = find_indicator_MinMax(train_df, indicators, indicators_period) # 指標極大極小表
                
                print('. . .stock id: ', portfoilo.iloc[period][j])
                evolution = population.run(train_df, moneyFortest[j], ta)
                evolution = pd.DataFrame.from_dict(evolution, orient="index")
                evolution.columns=['avg', 'std', 'best']
                evolution_route = '{}/evolution_{}.xlsx'.format(evolutionfolder, str(portfoilo.iloc[period][j]))
                evolution.to_excel(evolution_route)
                # get the best individual
                best = population.best()
                best_gene = population.getGene(best)
                kexpression = population.kexpression(best, ta)
                plotET(best, kexpression, (period+1), portfoilo.iloc[period][j], genefolder)
                # train period trade
                train_info = realTrade(chromosome = best, df = train_df, money=moneyFortrain[j], ta=ta)
                
                各期損益_訓練期.extend(train_info['trade_detail'])
                總交易損益_訓練期.extend(train_info['trade_detail'])
                各期損益MDD_訓練期.append(train_info['trade_date_detail'])
                # 各股交易資訊(訓練期)
                saveTradeDetail(route=train_period_folder_name,
                                stockID=portfoilo.iloc[period][j],
                                detail=train_info['trade_date_detail'],
                                performance=train_info)
                
                各期報酬率_訓練期.append(train_info['絕對報酬率'])
                moneyFortrain[j] += train_info['淨利'] # 更新目前本金(訓練)
                train_profit_ary.extend(train_info['trade_detail'])
                print('train: NT$', train_info['淨利'])
                
                
                # test period trade
                test_info = realTrade(chromosome = best, df = test_df, money=moneyFortest[j], ta=ta)
                
                各期損益_測試期.extend(test_info['trade_detail'])
                總交易損益_測試期.extend(test_info['trade_detail'])
                各期損益MDD_測試期.append(test_info['trade_date_detail'])
                # 各股交易資訊(測試期)
                saveTradeDetail(route=test_period_folder_name,
                                stockID=portfoilo.iloc[period][j],
                                detail=test_info['trade_date_detail'],
                                performance=test_info)
                
                各期報酬率_測試期.append(test_info['絕對報酬率'])
                moneyFortest[j] += test_info['淨利'] # 更新目前本金(測試)
                print('test: NT$', test_info['淨利'])
                print('\n')
                # "最高价": "high",
                # "最低价": "low",
                # "开盘价": "open",
                # "收盘价": "close",
                # "前收盘价": "adj close",
                # "成交量(千元)": "volume",
                # "交易时间": "date"
                drawdf = df.truncate(before = windows[s1], after = windows[e2])
                print(drawdf)
                drawTradePic(portfoilo.iloc[period][j],
                             drawdf[['open', 'close', 'low', 'high', 'volume']],
                             train_info['trade_date_detail'],
                             test_info['trade_date_detail'],
                             get_date(windows[s2]),
                             evolutionfolder,
                             int(period+1))   
            
            總交易各期損益_訓練期.append(sum(各期損益_訓練期))
            總交易各期損益_測試期.append(sum(各期損益_測試期))
            
            總交易各期損益最大交易回落訓練期 .extend(各期損益MDD_訓練期)
            總交易各期損益最大交易回落測試期 .extend(各期損益MDD_測試期)
            
            總交易報酬_訓練期.extend(各期報酬率_訓練期)
            總交易報酬_測試期.extend(各期報酬率_測試期)
            
            本金訓練期 = sum(moneyFortrain)-sum(各期損益_訓練期)
            本金測試期 = sum(moneyFortest)-sum(各期損益_測試期)
    
            train_profit_record[get_date(windows[s1])] = [ARR(各期報酬率_訓練期),
                                                          NetPorfit(各期損益_訓練期),
                                                          Odds(各期損益_訓練期),
                                                          TradeCount(各期損益_訓練期),
                                                          IRR(len(train_df), 各期損益_訓練期, 本金訓練期, 'daily'),
                                                          SharpRatio(len(train_df), 各期報酬率_訓練期, 'daily'),
                                                          MaxDrawdown(PeriodProfit(各期損益MDD_訓練期)),
                                                          MaxDrawdownRate(PeriodProfit(各期損益MDD_訓練期)),
                                                          RRR(各期損益_訓練期, MaxDrawdown(PeriodProfit(各期損益MDD_訓練期))),
                                                          ProfitFactor(各期損益_訓練期),
                                                          ProfitLossRatio(各期損益_訓練期),
                                                          Kelly(各期損益_訓練期)]
            
            test_profit_record[get_date(windows[s2])] = [ARR(各期報酬率_測試期), # 每期平均報酬
                                                         NetPorfit(各期損益_測試期), # 每期總淨利
                                                         Odds(各期損益_測試期), # 勝率
                                                         TradeCount(各期損益_測試期), # 交易次數
                                                         IRR(len(test_df), 各期損益_測試期, 本金測試期, 'daily'),
                                                         SharpRatio(len(test_df), 各期報酬率_測試期, 'daily'),
                                                         MaxDrawdown(PeriodProfit(各期損益MDD_測試期)),
                                                         MaxDrawdownRate(PeriodProfit(各期損益MDD_測試期)),
                                                         RRR(各期損益_測試期, MaxDrawdown(PeriodProfit(各期損益MDD_測試期))),
                                                         ProfitFactor(各期損益_測試期),
                                                         ProfitLossRatio(各期損益_測試期),
                                                         Kelly(各期損益_測試期)] 
    
            moneyFortrain  = [sum(moneyFortrain)/len(portfoilo.columns) for _ in range(len(portfoilo.columns))]
            moneyFortest  = [sum(moneyFortest)/len(portfoilo.columns) for _ in range(len(portfoilo.columns))]
    
            print('\n')
            new += 1
           #period += 1
        end = time.time()
        print('time cost', (end - start)/60, 'mins')
        
        train_performance = {'總淨利': sum(總交易各期損益_訓練期),
                            '絕對報酬率' : ROI(總交易各期損益_訓練期, 10000000),
                            '年化報酬': IRR(5, 總交易各期損益_訓練期, 10000000, 'year'),
                            '勝率': Odds(總交易損益_訓練期),
                            '最大交易回落': MaxDrawdown(PeriodProfit(總交易各期損益最大交易回落訓練期)),
                            '最大交易回落(%)': MaxDrawdownRate(PeriodProfit(總交易各期損益最大交易回落訓練期)),   
                            '風報比': RRR(總交易各期損益_訓練期, MaxDrawdown(PeriodProfit(總交易各期損益最大交易回落訓練期))),
                            '盈虧比': ProfitLossRatio(總交易各期損益_訓練期),
                            '獲利因子' : ProfitFactor(總交易各期損益_訓練期),
                            '凱利值' : Kelly(總交易損益_訓練期),
                            '夏普值(各期)': SharpRatio(2, [ary[0] for ary in train_profit_record.values()], 'season'),
                            '夏普值(總報)': SharpRatio(2, 總交易報酬_訓練期, 'season'),
                            '交易次數': len(總交易損益_訓練期)}
        
        test_performance = {'總淨利': sum(總交易各期損益_測試期),
                            '絕對報酬率': ROI(總交易各期損益_測試期, 10000000),
                            '年化報酬': IRR(5, 總交易各期損益_測試期, 10000000, 'year'),
                            '勝率': Odds(總交易損益_測試期),
                            '最大交易回落': MaxDrawdown(PeriodProfit(總交易各期損益最大交易回落測試期)),
                            '最大交易回落(%)': MaxDrawdownRate(PeriodProfit(總交易各期損益最大交易回落測試期)),
                            '風報比': RRR(總交易各期損益_測試期, MaxDrawdown(PeriodProfit(總交易各期損益最大交易回落測試期))),
                            '盈虧比': ProfitLossRatio(總交易各期損益_測試期),
                            '獲利因子' : ProfitFactor(總交易各期損益_測試期),
                            '凱利值' : Kelly(總交易損益_測試期),
                            '夏普值(各期)': SharpRatio(2, [ary[0] for ary in test_profit_record.values()], 'season'),
                            '夏普值(總報)': SharpRatio(2, 總交易報酬_測試期, 'season'),
                            '交易次數': len(總交易損益_測試期)}
        
        
        
        
        saveTotalTrade(route=filename + '/total_trade_history',
                       filename='train',
                       detail=train_profit_record,
                       performance=train_performance,
                       trade_detail = PeriodProfitDetail(總交易各期損益最大交易回落訓練期))
        saveTotalTrade(route=filename + '/total_trade_history',
                       filename='test',
                       detail=test_profit_record,
                       performance=test_performance,
                       trade_detail = PeriodProfitDetail(總交易各期損益最大交易回落測試期))

    
    