from gep2.node import *
from gep2.multigene import *
from gep2.tools import *
import copy
import random


class Individual:
    def __init__(self, genome, genetype):      
        self.genome = genome 
        self.genetype = genetype
        #self.decode = self.express(self.genome)
        
    def mutate(self, rate): 
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            if isinstance(gene, TradeGene):
                for i in range(len(gene.genome)):
                    selNode = gene.genome[i]
                    if isinstance(gene.genome[i], Node) and random.random() < rate:
                        if i == 0:
                            selNode = random.choice(gene.funcSet)
                        elif(i != 0 and i < gene.head):
                            if random.random() < 0.5:
                                selNode = random.choice(gene.funcSet)
                            else:
                                selNode = random.choice(gene.termSet)
                        gene.genome[i] = Node(selNode.name, selNode.arity)
        return type(self)(genes, self.genetype)

    # 反轉
    def invert(self): # only head gene
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            if gene.head > 1:
                gene.genome[1: gene.head] = reversed(gene.genome[1: gene.head])
        return type(self)(genes, self.genetype)
        
    def is_transpose(self):
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            head = gene.head
            allen = gene.allen
            if head > 1:
                pos = self.get_is_pos(gene.genome, head)
                if len(pos) > 0:
                    ispos = 0
                    start, end, head_pos = pos[0], pos[1], pos[2]
                    if end-start > head-head_pos:
                        for i in range(head-head_pos):
                            gene.genome[head_pos] = gene.genome[start: end][ispos]
                            head_pos += 1
                            ispos += 1
                    else:
                        for i in range(end-start):
                            gene.genome[head_pos] = gene.genome[start: end][ispos]
                            head_pos += 1
                            ispos += 1
        return type(self)(genes, self.genetype)
    
    def ris_transpose(self):
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            head = gene.head
            allen = gene.allen
            if head > 1:
                pos = self.get_ris_pos(gene.genome, head)
                if len(pos) > 0:
                    start = pos[0]
                    end = pos[1]
                    for i, v in zip(range(end-start), range(start, end)): 
                        del_g = gene.genome.pop(v)
                        gene.genome.insert(i, del_g)
        return type(self)(genes, self.genetype)
    
    # 單點重組
    def crossover_one_point(self, other):
        ind1, ind2 = copy.deepcopy(self.genome), copy.deepcopy(other.genome) # p
        for ind1_gene, ind2_gene in zip(ind1, ind2):
            if isinstance(ind1_gene, TradeGene) and isinstance(ind2_gene, TradeGene): 
                p1 = random.randint(0, len(ind1_gene.genome[:-2])-1) # random point on chro
                if p1 != len(ind1_gene.genome[:-2])-1:
                    ind1_gene.genome[p1:], ind2_gene.genome[p1:] = ind2_gene.genome[p1:], ind1_gene.genome[p1:]
        return type(self)(ind1, self.genetype), type(self)(ind2, self.genetype)
       
    # 雙點重組
    def crossover_two_point(self, other):
        ind1, ind2 = copy.deepcopy(self.genome), copy.deepcopy(other.genome)
        
        for ind1_gene, ind2_gene in zip(ind1, ind2):
            if isinstance(ind1_gene, TradeGene) and isinstance(ind2_gene, TradeGene):
                p1 = random.randint(0, len(ind1_gene.genome[:-2]))
                p2 = random.randint(p1+1, len(ind1_gene.genome[:-2])) if p1 != len(ind1_gene.genome[:-2]) else random.randint(0, p1-1)
                if p1 > p2: p1, p2 = p2, p1
                if (p2-p1) > 1:
                    ind1_gene.genome[p1: p2], ind2_gene.genome[p1: p2] = ind2_gene.genome[p1: p2], ind1_gene.genome[p1: p2]
        return type(self)(ind1, self.genetype), type(self)(ind2, self.genetype)

    def dc_mutate(self, rate):
        genes = copy.deepcopy(self.genome)
        genetype = self.genetype
        for gene in genes:
            if isinstance(gene, TradeGene):
                for node in gene.genome:
                    if isinstance(node, list):   
                        for dc, dctype in zip(node, gene.Dc):
                            gene_dc = dc[0]
                            gene_rnc = dc[1]
                            min_, max_ = dctype.threshold[0], dctype.threshold[1]
                            for dc_node in gene_dc:
                                dc_node = random.randint(0, len(gene_dc)-1)
                            for rnc_node in gene_rnc:
                                rnc_node = random.randint(min_, max_)
        return type(self)(genes, self.genetype)
            
    def dc_mutate(self, rate):
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            if isinstance(gene, TradeGene):
                for node in gene.genome:
                    if isinstance(node, list):   
                        for dc, dctype in zip(node, gene.Dc):
                            gene_dc = dc[0]
                            gene_rnc = dc[1]
                            min_, max_ = dctype.threshold[0], dctype.threshold[1]
                            for dc_node in gene_dc:
                                dc_node = random.randint(0, len(gene_dc)-1)
                            for rnc_node in gene_rnc:
                                rnc_node = random.randint(min_, max_)
            elif isinstance(gene, RiskGene):
                for node in gene.genome:
                    if isinstance(node, list):
                        for dc, dctype in zip(node, gene.Dc):
                            gene_dc = dc[0]
                            gene_rnc = dc[1]
                            min_, max_ = dctype.threshold[0], dctype.threshold[1]
                            for dc_node in gene_dc:
                                dc_node = random.randint(0, len(gene_dc)-1)
                            for rnc_node in gene_rnc:
                                rnc_node = random.randint(min_, max_)
            elif isinstance(gene, ZsorseGene):
                for node in gene.genome:
                    if isinstance(node, list):
                        for dc, dctype in zip(node, gene.Dc):
                            gene_dc = dc[0]
                            gene_rnc = dc[1]
                            min_, max_ = dctype.threshold[0], dctype.threshold[1]
                            for dc_node in gene_dc:
                                dc_node = random.randint(0, len(gene_dc)-1)
                            for rnc_node in gene_rnc:
                                rnc_node = random.randint(min_, max_)
        return type(self)(genes, self.genetype)
    
    
    def dc_invert(self):
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            for node in gene.genome:
                if isinstance(node, list):   
                    for dc, dctype in zip(node, gene.Dc):
                        gene_dc = dc[0]
                        dc_p1, dc_p2 = self.get_invert_pos(gene_dc)
                        gene_dc[dc_p1: dc_p2] = reversed(gene_dc[dc_p1: dc_p2])                           
        return type(self)(genes, self.genetype)
            
    def rnc_invert(self):
        genes = copy.deepcopy(self.genome)
        for gene in genes:
            for node in gene.genome:
                if isinstance(node, list):   
                    for rnc, dctype in zip(node, gene.Dc):
                        gene_rnc = rnc[1]
                        rnc_p1, rnc_p2 = self.get_invert_pos(gene_rnc)
                        gene_rnc[rnc_p1: rnc_p2] = reversed(gene_rnc[rnc_p1: rnc_p2])                           
        return type(self)(genes, self.genetype)
    
    def get_invert_pos(self, ary):
        p1 = random.randint(0, len(ary))
        p2 = random.randint(p1+1, len(ary)) if p1 != len(ary) else random.randint(0, p1-1)
        if p1 > p2: p1, p2 = p2, p1
        return p1, p2
    
    def get_is_pos(self, gene, head):
        geneLen = 0
        for node in gene:
            if isinstance(node, Node):
                geneLen += 1
        g_p1 = random.randint(1, geneLen) # IS插入起始
        g_p2 = random.randint(g_p1+1, geneLen) if g_p1 != geneLen else random.randint(1, g_p1-1) # IS插入終點
        h_p1 = random.randint(1, head-1) # 頭部插入起始
        if g_p1 > g_p2: g_p1, g_p2 = g_p2, g_p1
        return [g_p1, g_p2, h_p1]
    
    def get_ris_pos(self, gene, head):
        gene = gene[0: -2]
        start = random.randint(1, head-1) # 起始位置位於頭部且根結點不能作為起始點
        while gene[start].arity < 1 and start != head-1:
            start += 1
        if start == head-1 and gene[start].arity < 1:
            return []
        else:
            end = random.randint(start+1, head)
            return [start, end]
    
    
    
    
    
    
    
    
