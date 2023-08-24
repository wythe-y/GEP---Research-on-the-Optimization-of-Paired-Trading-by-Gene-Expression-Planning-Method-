from paper2.gep2.node import *
from paper2.gep2.primitiveset import *
from paper2.gep2.multigene import *
from paper2.gep2.chromosome import *
from paper2.gep2.population import *
from paper2.gep2.stockdata import *
from paper2.gep2.tools import *


def get_windows(year, start_year):
    season = ['-03-31', '-05-15', '-08-14', '-11-14']
    year = year[year.index(start_year):-1]
    windows = []
    for y in year:
        for s in season:
            w = str(y) + s
            windows.append(w)
    return windows


if __name__ == '__main__':
        
    indicators = ['K', 'D', 'RSI', 'BIAS', 'WILLR', 'MOM', 'AR']
    period = [5, 60] # 指標天期
    stock = 2330
    
    df = get_stock_data(sid=stock, start='2010-01-01', end='2021-03-31',
                        indicators=indicators, period=period)
    df = df.dropna()
    df = df[df.index.year>2011]
    year = sorted(set(df.index.year))
    windows = get_windows(year = year, start_year = 2015)

    train_window_size = 4 # 6m
    test_window_size = 3 # 3m
    slide_window_size = test_window_size # 3m
      
    moneyFortrain, moneyFortest = 5000000, 5000000  # 訓練與測試用資金
    test_profit_record = dict() # 紀錄測試期資金
    
    
    ''' 基因1 2 '''
    
    lot = [1, 5] # 買賣張數
    trade_gene = Gene(name='TradeGene',
                      pset=set_trade_gene(indicators, period, lot),
                      head=6)
    
    dc_period = GeneDc(dcName='Period',
                       dcLen=trade_gene.head,
                       rncLen=trade_gene.head*2,
                       threshold=period)
    
    dc_threshold_min = GeneDc(dcName='Threshold_min',
                          dcLen=trade_gene.head,
                          rncLen=trade_gene.head*2,
                          threshold=[0, 1])
    
    dc_threshold_max = GeneDc(dcName='Threshold_max',
                          dcLen=trade_gene.head,
                          rncLen=trade_gene.head*2,
                          threshold=[0, 1])
    
    # 由於頭部長度會縮減因此買賣張數dc為 尾部長度 + 頭部長度 - 1
    dc_Lot = GeneDc(dcName='Lot',
                    dcLen=trade_gene.tail + trade_gene.head - 1,
                    rncLen=trade_gene.tail*2,
                    threshold=lot)
    
    trade_gene.add_dc_rnc([dc_period, dc_threshold_min, dc_threshold_max, dc_Lot])
    
    
    ''' 基因3 '''
    risk_threshold = [1, 30] # 停損停利閥值 %
    risk_gene = Gene(name='RiskGene',
                     pset=set_risk_gene(risk_threshold),
                     head=0)
    
    dc_risk = GeneDc(dcName='Risk',
                     dcLen=risk_gene.tail,
                     rncLen=risk_gene.tail*2,
                     threshold=risk_threshold)
    risk_gene.add_dc_rnc([dc_risk])
    
    
    ''' 染色體 '''
    chro = Chromosome([trade_gene, risk_gene])

    
    trade_perid = 1
    for i in range(0, len(windows)-slide_window_size-1, slide_window_size):
        print('period: ', trade_perid)
        train_period = i + train_window_size
        test_period = train_period  + test_window_size
        
        
        #train_start, train_end = windows[i: train_period][0], windows[i: train_period][-1]
        #if len(windows[train_period: test_period])
        #test_start, test_end = windows[i: train_period][0], windows[train_period: test_period][0]
        #print(windows[i: train_period])
        #print(windows[train_period: test_period])

        
        # train data
        train_start, train_end = windows[i: train_period][0], windows[i: train_period][-1]
        train_df = df.truncate(before=train_start, after=train_end)
        print('train start', train_start, 'train end', train_end, 'len', len(train_df))
        # test data
        test_end = windows[train_period: test_period][0] if len(windows[train_period: test_period]) > 0 else str(year[-1]) + '-03-31'
        test_df = df.truncate(before=train_end, after=test_end)
        #print('train len', len(train_df)/20)
        print('test start', train_end, 'test end', test_end, 'len', len(test_df))
        len_ = len(test_df)/(len(test_df) + len(train_df))
        #print(round(len_, 2))
        #print('*' * 20)
        
        '''
        population = Population(chro=chro.genes, popSize=30, popIter=20)
        
        ta = find_indicator_MinMax(train_df, indicators, period)
        # training
        population.run(train_df, moneyFortest, ta)
        
        # get the best individual
        best = population.best()
        
        # train period trade
        train_info = realTrade(chromosome = best, df = train_df, money=moneyFortrain, ta=ta)
        #trade_train_history[str(year[train])+'~train'] = trade_detail
        moneyFortrain += train_info['net_profit']
        
        # tesr period trade
        test_info = realTrade(chromosome = best, df = test_df, money=moneyFortest, ta=ta)         
        #trade_test_history[str(year[test])+'~test'] = trade_detail
        moneyFortest += test_info['net_profit']
        test_profit_record[trade_perid] = [len_, test_info['net_profit']]
        #test_profit_record.append(test_info['net_profit'])
        trade_perid += 1'''
        
    

    



    
    

    
    
    
   

















