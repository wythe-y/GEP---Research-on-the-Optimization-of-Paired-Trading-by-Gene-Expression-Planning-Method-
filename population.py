from gep2.chromosome import *
from gep2.tools import *
from gep2.individual import *
from gep2.tools import *
from gep2.fitness import *
import copy 
import numpy as np

class Population:
    def __init__(self, chro, popSize, popIter):
        self._chro = chro
        self._size = popSize
        self._iter = popIter
        self.mutation_rate = 0.3
        self.inversion_rate = 0.2
        self.is_transposition_rate = 0.2
        self.ris_transposition_rate  = 0.2
        self.crossover_one_point_rate = 0.4
        self.crossover_two_point_rate = 0.4
        self.dc_mutation_rate = 0.3
        self.dc_inversion_rate = 0.3
        self.dc_transposition_rate = 0.3
        self.rnc_inversion_rate = 0.2
        self.fitness = None
    
    def initialize(self, num=0):
        size = num if num else self._size
        genes = self._chro
        self.individual = [Individual(update_genome(generate(self._chro)), self._chro) for _ in range(size)]
    
    def evaluate(self, df, moneyFortrain, ta):
        result = dict()
        for i in range(len(self.individual)):
            info = fitness(self.individual[i].genome, df, moneyFortrain, ta)
            result[i] = {'gene': self.individual[i],
                         'ffunc': info['RRR']}
        result = sorted(result.items(), key=lambda i:(i[1]['ffunc']), reverse=True)
        self.fitness = result
        fitness_avg = sum([f[1]['ffunc'] for f in dict(self.fitness).items()])/len(result)
        std = np.std([f[1]['ffunc'] for f in dict(self.fitness).items()])
        best = max([f[1]['ffunc'] for f in dict(self.fitness).items()])
       # print('1. evaluate end')
        return {'fitness_avg': round(fitness_avg, 2), 'std': round(std, 2), 'best': round(best, 2)}
        
    def selection(self):
        # roulette wheel selection
        fitness = self.fitness # fitness dict include gene & fitness
        size = self._size # will select pop's num
        # 菁英法 本代中的取前三名強制放入下一代中
        elitism_size = 1 # 菁英數量
        sel_gene = [f['gene'] for f in dict(fitness[: elitism_size]).values()]      
        # 輪盤法
        fitness = dict(fitness)
        ind_id = [f[0] for f in fitness.items()]
        fitness_val = [f[1]['ffunc'] for f in fitness.items()]      
        fitness_min = min(fitness_val) # find minimum in list and check whether any negative val in list 
        if fitness_min < 0: # 如果適應值中有負值，加上最小值使整個適應度為正值
            fitness_val = [f + abs(fitness_min) for f in fitness_val] # cover negative val
        sum_fit = sum(fitness_val)

        if sum_fit > 0:
            prob = [f/sum_fit for f in fitness_val] # calculate each individual prob
            sel_id = np.random.choice(ind_id, int((size-elitism_size) * 0.1), True, prob)
            #sel_gene = [fitness[key]['gene'] for key in sel_id]
            sel_gene.extend([fitness[key]['gene'] for key in sel_id])
        else: # 整個族群中每個染色體都沒有進行交易時
            sel_gene = None
        #print('2. selection end')
        return sel_gene
       
    def best(self):
        fitness = self.fitness # fitness dict include gene & fitness
        # rr top 1
        best_ind = dict(fitness[: 1])
        return list(best_ind.values())[0]['gene'].genome

    def kexpression(self, genome, ta):
        bestGene = dict()
        trade_gene = list()
        risk_gene = list()
        zsorse_gene = list()        
        genes = genome
        for gene in genes:
            if isinstance(gene, TradeGene):
                dc_index = 0
                dc_lot_index = 0
                # 指標天期
                dc_period = gene.genome[-1][0][0]
                rnc_period = gene.genome[-1][0][1]
                # 指標閥值
                dc_min_ = gene.genome[-1][1][0]
                rnc_min_ = gene.genome[-1][1][1]
                dc_max_ = gene.genome[-1][2][0]
                rnc_max_ = gene.genome[-1][2][1]
                # 買賣張數
                dc_lot = gene.genome[-1][3][0]
                rnc_lot = gene.genome[-1][3][1]
                for node in gene.genome: 
                    if isinstance(node, Node) and node.arity>0:
                        print(dc_period,rnc_period,dc_min_,rnc_min_,dc_max_,rnc_max_,dc_lot,rnc_lot)
                        period = rnc_period[dc_period[dc_index]] # 天期
                        min_, max_ = rnc_min_[dc_min_[dc_index]], rnc_max_[dc_max_[dc_index]] #最小最大
                        if min_ > max_:
                            min_, max_ = max_, min_
                        taRange = ta[node.name][1] - ta[node.name][0]
                        min_ = int(float(min_ * taRange + min_))
                        max_ = int(float(max_ * taRange + max_))
                        nodeName = '{}_{},{},{}'.format(node.name, period, min_, max_)
                        dc_index += 1
                    elif isinstance(node, Node) and node.arity == 0: 
                        lot = rnc_lot[dc_lot[dc_lot_index]]
                        if node.name == 1:
                            nodeName = 'buy_{}'.format(lot)
                        elif node.name == 0:
                            nodeName = 'no action'
                        elif node.name == -1:
                            nodeName = 'sell_{}'.format(lot)
                        dc_lot_index += 1
                    trade_gene.append(nodeName)
            elif isinstance(gene, ZsorseGene):
                dc_index = 0
               #dc_lot_index = 0
                dc_zsorseperiodz = gene.genome[-1][0][0]
                rnc_zsorseperiodz = gene.genome[-1][0][1]
                dc_thresholdz_min = gene.genome[-1][1][0]
                rnc_thresholdz_min = gene.genome[-1][1][1]
                dc_thresholdz_max = gene.genome[-1][2][0]
                rnc_thresholdz_max = gene.genome[-1][2][1]
                for node in gene.genome: 
                    if isinstance(node, Node) and node.arity>0:
                        if dc_index < len(dc_zsorseperiodz) and \
                            dc_index < len(dc_thresholdz_min) and dc_index < len(dc_thresholdz_max) and \
                            dc_thresholdz_min[dc_index] < len(rnc_thresholdz_min) and \
                            rnc_thresholdz_max[dc_index] < len(dc_thresholdz_max):
                
                            period = rnc_zsorseperiodz[dc_zsorseperiodz[dc_index]] # 天期
                            min_, max_ = rnc_thresholdz_min[dc_thresholdz_min[dc_index]], rnc_thresholdz_max[dc_thresholdz_max[dc_index]] #最小最大
                            if min_ > max_:
                                min_, max_ = max_, min_
                            taRange = ta[node.name][1] - ta[node.name][0]
                            min_ = int(float(min_ * taRange + min_))
                            max_ = int(float(max_ * taRange + max_))
                            nodeName = '{}_{},{},{}'.format(node.name, period, min_, max_)
                            dc_index += 1

                # for node in gene.genome: 
                #     if isinstance(node, Node) and node.arity>0:
                #         print(dc_zsorseperiodz,rnc_zsorseperiodz,rnc_thresholdz_min,dc_index,dc_zsorseperiodz,dc_thresholdz_min,dc_thresholdz_max,rnc_thresholdz_max)
                #         period = rnc_zsorseperiodz[dc_zsorseperiodz[dc_index]] # 天期
                #         min_, max_ = rnc_thresholdz_min[dc_thresholdz_min[dc_index]], dc_thresholdz_max[rnc_thresholdz_max[dc_index]] #最小最大
                #         if min_ > max_:
                #             min_, max_ = max_, min_
                #         taRange = ta[node.name][1] - ta[node.name][0]
                #         min_ = int(float(min_ * taRange + min_))
                #         max_ = int(float(max_ * taRange + max_))
                #         nodeName = '{}_{},{},{}'.format(node.name, period, min_, max_)
                #         dc_index += 1
                    elif isinstance(node, Node) and node.arity == 0: 
                       #lot = rnc_lot[dc_lot[dc_lot_index]]
                        if node.name == 1:
                            nodeName = 'buy_{}'.format(lot)
                        elif node.name == 0:
                            nodeName = 'no action'
                        elif node.name == -1:
                            nodeName = 'sell_{}'.format(lot)
                        dc_lot_index += 1
                    zsorse_gene.append(nodeName)
            elif isinstance(gene, RiskGene):
                dc = gene.genome[-1][0][0]
                rnc = gene.genome[-1][0][1]
                risk_gene = '{}%,{}%'.format(round(rnc[dc[0]], 2),round(rnc[dc[1]], 2)) # stop loss/ stop profit
            else:
                print('this gene is not find')
            bestGene['trade'] = trade_gene
            bestGene['risk'] = risk_gene
            bestGene['zsorse'] = zsorse_gene
        return bestGene


    def getGene(self, genome):
        bestGene = dict()
        trade_gene = list()
        risk_gene = list()
        zsorse_gene = list() 
        best_gene = genome
        for node in best_gene[0].genome:
            current_node = ''
            if isinstance(node, Node):
                if node.name == 1:
                    current_node = 'buy'
                elif node.name == -1:
                    current_node = 'sell'
                elif node.name == 0:
                    current_node = 'no action'
                else:
                    current_node = node.name
            elif isinstance(node, list):
                current_node = node
            trade_gene.append(current_node)
                              
        for node in best_gene[1].genome:
            current_node = ''
            if isinstance(node, Node):
                current_node = node.name
            elif isinstance(node, list):
                current_node = node
            risk_gene.append(current_node)
             
        for node in best_gene[2].genome:
            current_node = ''
            if isinstance(node, Node):
                if node.name == 1:
                    current_node = 'buy'
                elif node.name == -1:
                    current_node = 'sell'
                elif node.name == 0:
                    current_node = 'no action'
                else:
                    current_node = node.name
            elif isinstance(node, list):
                current_node = node
            zsorse_gene.append(current_node)
            
        bestGene['trade'] = trade_gene
        bestGene['risk'] = risk_gene
        bestGene['zsorse'] = zsorse_gene
        return bestGene


    def run(self, df, money, ta):
        #print("{:<6} {:<10} {:<10} {:<10}".format('iter','avg', 'std', 'best'))
        evaluateDict = dict()
        self.initialize(self._size) # 初始化族群
        for i in range(self._iter):
            #self.evaluate(df, money, ta)     
            result_dict = self.evaluate(df, money, ta)     
            avg = result_dict['fitness_avg']
            std = result_dict['std']
            best = result_dict['best']
            evaluateDict[i] = [avg, std, best]
            #print("{:<6} {:<10} {:<10} {:<10}".format(i+1, avg, std, best))
            # 選擇
            self.seleted_gene = self.selection()
            # 下一代
            self.next_pop = []
            if self.seleted_gene != None:
                for ind in self.seleted_gene:
                    
                    # 突變
                    ind_mutate  = ind.mutate(self.mutation_rate)
                    ind_mutate.genome = update_genome(ind_mutate.genome)
                    self.next_pop.append(ind_mutate)
                
                    # 反轉
                    if random.random() < self.inversion_rate:
                        ind_invert = ind.invert()
                        ind_invert.genome = update_genome(ind_invert.genome)
                        self.next_pop.append(ind_invert)
                    
                    '''
                    # 轉置
                    if random.random() < self.is_transposition_rate:
                        ind_is = ind.is_transpose()
                        ind_is.genome = update_genome(ind_is.genome)
                        self.next_pop.append(ind_is)'''
                        
                    # 根轉置
                    if random.random() < self.ris_transposition_rate:
                        ind_ris = ind.ris_transpose()
                        ind_ris.genome = update_genome(ind_ris.genome)
                        self.next_pop.append(ind_ris) 
                    
                    
                    # 單點交換
                    for i, j in self.crossover_pair(self.crossover_one_point_rate):
                        par1, par2 = self.seleted_gene[i], self.seleted_gene[j]
                        child1, child2 = par1.crossover_one_point(par2)
                        child1.genome = update_genome(child1.genome)
                        child2.genome = update_genome(child2.genome)
                        self.next_pop.append(child1)
                        self.next_pop.append(child2)
                        
                    # 雙點交換
                    for i, j in self.crossover_pair(self.crossover_two_point_rate):
                        par1, par2 = self.seleted_gene[i], self.seleted_gene[j]
                        child1, child2 = par1.crossover_two_point(par2)
                        child1.genome = update_genome(child1.genome)
                        child2.genome = update_genome(child2.genome)
                        self.next_pop.append(child1)
                        self.next_pop.append(child2)
                    
                    for ind in self.seleted_gene:
                        # dc rnc突變
                        dc_mutate = ind.dc_mutate(self.dc_mutation_rate)
                        dc_mutate.genome = update_genome(dc_mutate.genome)
                        self.next_pop.append(dc_mutate)
                        
                        # dc反轉
                        if random.random() < self.dc_inversion_rate:
                            dc_invert = ind.dc_invert()
                            dc_invert.genome = update_genome(dc_invert.genome)
                            self.next_pop.append(dc_invert)
                        
                        # rnc反轉
                        if random.random() < self.dc_inversion_rate:
                            rnc_invert = ind.rnc_invert()
                            rnc_invert.genome = update_genome(rnc_invert.genome)
                            self.next_pop.append(rnc_invert)
                            
                self.next_pop.extend(self.seleted_gene)
                if len(self.next_pop) < self._size:              
                    newgene = [Individual(update_genome(generate(self._chro)), self._chro) for _ in range(self._size-len(self.next_pop))]
                    self.next_pop.extend(newgene)
                setattr(self, 'individual', self.next_pop)
            else:
                self.initialize()
        return evaluateDict
            
            
    def crossover_pair(self, rate):
        orgs = [i for i in range(len(self.seleted_gene)) if random.random() < rate]
        pair = []
        if len(orgs) > 3:
            random.shuffle(orgs)
            for i in range(0, len(orgs), 2):
                try:
                    pair.append([orgs[i], orgs[i+1]])
                except:
                    pair.append([orgs[i], random.choice(orgs[:-1])])
        return pair
        
        
    