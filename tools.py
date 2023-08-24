from gep2.primitiveset import *
from gep2.multigene import TradeGene,RiskGene,ZsorseGene
import random
import datetime
import pandas as pd
import numpy as np
import copy
import mplfinance as mpf
import xlsxwriter
from graphviz import Digraph


def get_windows(year, start_year):
    season = ['0331', '0515', '0814', '1114']
    year = year[year.index(start_year):]
    windows = []
    for y in year:
        for s in season:
            w = str(y) + s
            windows.append(w)
    return windows

def get_date(str_): # 更新
    dt = datetime.datetime.strptime(str_, "%Y%m%d")
    dt = dt + datetime.timedelta(days=1)
    #print(str(dt).split()[0].replace('-', ''))
    return str(dt).split()[0].replace('-', '')

# def get_date(pd_timestamp):
#     pd_timestamp = pd_timestamp + pd.Timedelta(days=1)
#     dt_str = pd_timestamp.strftime("%Y%m%d")
#     print(dt_str)
#     return dt_str

def find_indicator_MinMax(data, indicators, period):
    d = dict()
    per_min = period[0] # 最小天期
    per_max = period[1] + 1 # 最大天期
    for i in indicators:
        cols = list()
        for p in range(per_min, per_max):
            col = '{}_{}'.format(i, p)
            cols.append(col)
        max_ = max(data[cols].max())
        min_ = min(data[cols].min())
        d[i] = [min_, max_]
   #print(d)
    return d

def set_trade_gene(indicators, period, lot):
    tg_pset = PrimitiveSet('tg_pset')
    # 添加符號
    indicator = indicators
    action = [1,-1, 0] # buy, sell, no action 
    #action = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tg_pset.add_function(indicator, 3)
    tg_pset.add_terminal(action)
    # 添加閥值 指標天期、買賣張數
    period = period
    lot = lot
    
    rnc_ary = []
    rnc1 = RncForFunc(period)
    rnc_ary.append(rnc1)
    
    rnc2 = RncForTerm(lot)
    rnc_ary.append(rnc2)
    
    tg_pset.add_rnc_threshold(rnc_ary)
    return tg_pset

def set_zsorse_gene(zsorse, period):
    zs_pset = PrimitiveSet('zs_pset')
    # 添加符號
    zsorse = zsorse
    action = [1,-1, 0] # buy, sell, no action 
    #action = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    zs_pset.add_function(zsorse, 3)
    zs_pset.add_terminal(action)
    
    # 添加閾值
    rnc_ary = []
    rnc1 = RncForFunc(period)
    rnc_ary.append(rnc1)
    
    zs_pset.add_rnc_threshold(rnc_ary)
    
    return zs_pset

def set_fundmgt_gene():
    fg_pset = PrimitiveSet('fg_pset')
    return fg_pset

def set_risk_gene(threshold):
    rg_pset = PrimitiveSet('rg_pset')
    # 添加符號
    rg_pset.add_rnc_terminal()
    # 添加閥值 停損停利
    rnc_ary = []
    rnc_ary.append(RncForTerm(threshold))
    rg_pset.add_rnc_threshold(rnc_ary)
    return rg_pset


def generate(genes):
    genome = []
    for gene in genes:
        funcSet = gene.pset.functions 
        termSet = gene.pset.terminals 
        allen = gene.allen
        this_genome = [None] * allen
        dc_ary = []
        if gene.name == 'TradeGene' or gene.name == 'FundMgtGene':
            for i in range(gene.allen):
                selNode = 0
                if i < gene.head:
                    if i == 0: # root node must be a func
                        selNode = random.choice(funcSet)
                    else: # head can be func or term node
                        selNode = random.choice(funcSet) if random.random() < 0.5 else random.choice(termSet)      
                else:
                    selNode = random.choice(termSet)
                this_genome[i] = Node(selNode.name, selNode.arity)
            dc = generateDc(gene.Dc)
            this_genome.append(dc)
            this_ = TradeGene(this_genome, gene)
            genome.append(this_)
        elif gene.name == 'Z-sorseGene':
            for i in range(gene.allen):
                selNode = 0
                if i < gene.head:
                    if i == 0: # root node must be a func
                        selNode = random.choice(funcSet)
                    else: # head can be func or term node
                        selNode = random.choice(funcSet) if random.random() < 0.5 else random.choice(termSet)      
                else:
                    selNode = random.choice(termSet)
                this_genome[i] = Node(selNode.name, selNode.arity)
            dc = generateDc(gene.Dc)
            this_genome.append(dc)
            this_ = ZsorseGene(this_genome, gene)
            genome.append(this_)
        elif gene.name == 'RiskGene':
            for i in range(gene.allen):
                selNode = random.choice(termSet)
                this_genome[i] = Node(selNode.name, selNode.arity)
            dc = generateDc(gene.Dc)
            this_genome.append(dc)
            this_ = RiskGene(this_genome, gene)
            genome.append(this_)
    return genome

def generateDc(geneDc):
    dc_ary = []
    for dc in geneDc:
        min_, max_ = dc.threshold[0], dc.threshold[1]
        if max_ - min_ != 1:           
            gene_dc = [random.randint(0, dc.rncLen-1) for _ in range(dc.dcLen)]
            gene_rnc = [random.randint(min_, max_) for _ in range(dc.rncLen)]
            this_ = [gene_dc, gene_rnc]
            dc_ary.append(this_)
        else:
            gene_dc = [random.randint(0, dc.rncLen-1) for _ in range(dc.dcLen)]
            gene_rnc = [round(random.random(), 5) for _ in range(dc.rncLen)]
            this_ = [gene_dc, gene_rnc]
            dc_ary.append(this_)
    return dc_ary

def update_genome(genome): # 填入rnc
    for gene in genome:
        if isinstance(gene, TradeGene):#zscore和trader同时被出现
            dc_index = 0
            dc_lot_index = 0
            # 指標天期
            #print(gene.genome)
            if len(gene.genome[-1]) == 3: #zcore
                dc_zsorseperiodz = gene.genome[-1][0][0]
                rnc_zsorseperiodz = gene.genome[-1][0][1]
                dc_thresholdz_min = gene.genome[-1][1][0]
                rnc_thresholdz_min = gene.genome[-1][1][1]
                dc_thresholdz_max = gene.genome[-1][2][0]
                rnc_thresholdz_max = gene.genome[-1][2][1]
            elif len(gene.genome[-1]) == 4: #tader
                #天期
                dc_period = gene.genome[-1][0][0]
                rnc_period = gene.genome[-1][0][1]
                # 指標閥值
                dc_min_ = gene.genome[-1][1][0]
                rnc_min_ = gene.genome[-1][1][1]
                dc_max_ = gene.genome[-1][2][0]
                rnc_max_ = gene.genome[-1][2][1]
                # # 買賣張數                
                dc_lot = gene.genome[-1][3][0]
                rnc_lot = gene.genome[-1][3][1]


            for node in gene.genome: 
                if isinstance(node, Node) and node.arity>0:
                    period = rnc_period[dc_period[dc_index]]
                    node.update_period(period)
                    min_, max_ = rnc_min_[dc_min_[dc_index]], rnc_max_[dc_max_[dc_index]]
                    node.update_threshsold(min_, max_)
                    node.update_lot(None)
                    dc_index += 1
                elif isinstance(node, Node) and node.arity == 0: 
                    #print(N)
                    #print(rnc_lot[dc_lot[dc_lot_index]])
                    lot = rnc_lot[dc_lot[dc_lot_index]]
                    node.update_lot(lot)
                    node.update_period(None)
                    node.update_threshsold(None, None)
                    dc_lot_index += 1
            update_child(gene.genome)
        if isinstance(gene, ZsorseGene):#zscore和trader同时被出现
            dc_index = 0
            dc_lot_index = 0
            # 指標天期
            #print(gene.genome)
            if len(gene.genome[-1]) == 3: #zcore
                dc_zsorseperiodz = gene.genome[-1][0][0]
                rnc_zsorseperiodz = gene.genome[-1][0][1]
                dc_thresholdz_min = gene.genome[-1][1][0]
                rnc_thresholdz_min = gene.genome[-1][1][1]
                dc_thresholdz_max = gene.genome[-1][2][0]
                rnc_thresholdz_max = gene.genome[-1][2][1]
            elif len(gene.genome[-1]) == 4: #tader
                #天期
                dc_period = gene.genome[-1][0][0]
                rnc_period = gene.genome[-1][0][1]
                # 指標閥值
                dc_min_ = gene.genome[-1][1][0]
                rnc_min_ = gene.genome[-1][1][1]
                dc_max_ = gene.genome[-1][2][0]
                rnc_max_ = gene.genome[-1][2][1]
                # # 買賣張數                
                dc_lot = gene.genome[-1][3][0]
                rnc_lot = gene.genome[-1][3][1]


            for node in gene.genome: 
                if isinstance(node, Node) and node.arity>0:
                    period = rnc_zsorseperiodz[dc_zsorseperiodz[dc_index]]
                    node.update_period(period)
                    min_, max_ = rnc_thresholdz_min[dc_thresholdz_min[dc_index]], rnc_thresholdz_max[dc_thresholdz_max[dc_index]]
                    node.update_threshsold(min_, max_)
                    node.update_lot(None)
                    dc_index += 1
                elif isinstance(node, Node) and node.arity == 0: 
                    #print(N)
                    #print(rnc_lot[dc_lot[dc_lot_index]])
                   #lot = rnc_lot[dc_lot[dc_lot_index]]
                   #node.update_lot(lot)
                    node.update_period(None)
                    node.update_threshsold(None, None)
                   #dc_lot_index += 1
            update_child(gene.genome)
        elif isinstance(gene, RiskGene):
            dc = gene.genome[-1][0][0]
            rnc = gene.genome[-1][0][1]
            gene.genome[0].update_risk(rnc[dc[0]])
            gene.genome[1].update_risk(rnc[dc[1]])
    # print(dc_period,rnc_period,dc_min_,rnc_min_,dc_max_,rnc_max_,dc_lot,rnc_lot)
    # print(dc_zsorseperiodz,rnc_zsorseperiodz,rnc_thresholdz_min,dc_index,dc_zsorseperiodz,dc_thresholdz_min,dc_thresholdz_max,rnc_thresholdz_max)
    return genome
    
def update_child(genome):
    i = 0
    j = 1
    while i < len(genome):
        if isinstance(genome[i], Node):
            if genome[i].arity > 0:
                child = list()
                for _ in range(genome[i].arity):
                    child.append(genome[j])
                    j += 1
                genome[i].update_child(child)
            else:
                genome[i].update_child(None)
        i += 1


def plotET(best_gene, kexpression, period, stock, route):
    tradeGene = best_gene[0].genome
    tradeGeneInfo = kexpression['trade']
    riskGeneInfo = kexpression['risk']
    nodes, labels, edges, edgesInfo = get_graph_kexpression(tradeGene, tradeGeneInfo)
    dot = Digraph(comment='The Round Table')
    for i in range(len(nodes)):
        dot.node(str(nodes[i]), str(labels[i]))
    for e in range(len(edges)):
        dot.edge(edges[e][0], edges[e][1], label=edgesInfo[e])
    stoploss, stopprofit = riskGeneInfo.split(',')[0], riskGeneInfo.split(',')[1]
    dot.node('risk', 'stop loss: {}\n stop profit: {}'.format(stoploss, stopprofit), shape = "box")  
    fname = '{}/period{}_{}_expressiontree'.format(route, period, stock)
    dot.render(fname, view=False)

def get_graph_kexpression(gene, info):
    i = 0
    j = 1
    nodes = ['0']
    labels = [info[0].split(',')[0]]
    edges = list()
    edgesInfo = list()
    while i < len(gene):
        if isinstance(gene[i], Node):
            if gene[i].arity > 0:
                min_ = info[i].split(',')[1]
                max_ = info[i].split(',')[2]
                edgesInfo.append('<{}'.format(min_))
                if min_ == max_:
                  edgesInfo.append('={}'.format(min_, max_))  
                else:
                  edgesInfo.append('{}-{}'.format(min_, max_))  
                edgesInfo.append('>{}'.format(max_))
                for _ in range(gene[i].arity):
                    nodes.append(str(j))
                    if gene[j].arity > 0:
                        labels.append(info[j].split(',')[0])
                    else:
                        labels.append(info[j])
                    edges.append((str(i), str(j)))
                    j += 1
        i += 1
    
    return nodes, labels, edges, edgesInfo#, arity
    
def drawTradePic(stockid, stockdata, trainDetail, testDetail, split_time, evolutionfolder, period):
    stock = stockid
    df = stockdata.copy()
    df.columns = ['Open', 'Close', 'Low', 'High', 'Volume']
    trainDetail = pd.DataFrame.from_dict(trainDetail, orient='index').rename(columns={0: 'price',
                                                                                      1: 'action',
                                                                                      2: 'lot',
                                                                                      3: 'profit'})
    #trainDetail.index = pd.DatetimeIndex(trainDetail.index)
    testDetail = pd.DataFrame.from_dict(testDetail, orient='index').rename(columns={0: 'price',
                                                                                    1: 'action',
                                                                                    2: 'lot',
                                                                                    3: 'profit'})
    #testDetail.index = pd.DatetimeIndex(testDetail.index)
    trade_all_detail = pd.concat([trainDetail, testDetail], axis=0)
    #print(trade_all_detail.columns)
    df = df.merge(trade_all_detail, how='outer', left_index=True, right_index=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    # change kbar type: up to red, down for green
    df['MCOverrides'] = np.where(df.Close > df.Open, 'red', 'green') 
    split_time = datetime.datetime.strptime(split_time, '%Y%m%d')
    df['trainTime'] = np.where(df.index < split_time, ((df.High+df.Low)/2), np.NaN)
    df['testTime'] = np.where(df.index >= split_time, ((df.High+df.Low)/2), np.NaN)
    if len(trade_all_detail.columns) > 0:
        # buy point
        df['buysignal'] = np.where(df.action == 'buy', df.Low*0.95, np.NaN) 
    
        # sell point
        df['sellsignal'] = np.where((df.action=='sell') |
                                    (df.action=='stop profit stop loss') |
                                    (df.action=='time up sell all'), 
                                     df.High*1.05, np.NaN)


        apds = [mpf.make_addplot(df.buysignal, type='scatter',markersize=200, marker='^', alpha=0.8),
                mpf.make_addplot(df.sellsignal, type='scatter',markersize=200, marker='v', alpha=0.8),
                mpf.make_addplot(df.trainTime, type='line', color='g', width=30, alpha=0.1),
                mpf.make_addplot(df.testTime, type='line', color='b', width=30, alpha=0.1)]
    else:
        apds = [mpf.make_addplot(df.trainTime, type='line', color='g', width=30, alpha=0.1),
                mpf.make_addplot(df.testTime, type='line', color='b', width=30, alpha=0.1)]
    mpf.plot(df, figratio=(50,20), style='yahoo', title='\n\n{}.TW'.format(stock),
             type='candle', marketcolor_overrides=df.MCOverrides.values,
             addplot=apds, savefig='{}/period_{}_{}.png'.format(evolutionfolder, period, stock))
  
