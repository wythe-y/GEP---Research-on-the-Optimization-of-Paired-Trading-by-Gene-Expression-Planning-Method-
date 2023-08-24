from gep2.node import *
from gep2.tools import *

class Gene: # gene rule for generate
    def __init__(self, name, pset, head=0):
        self._name = name
        self._pset = pset
        self._head = head
        self._arity = pset.max_arity if head else 0
        self._tail = head * (self._arity - 1) + 1 if head else 2 
        self._allen = self._head + self._tail
        self._dc = []
        
    def add_dc_rnc(self, ary):
        setattr(self, '_dc', ary)
            
    @property
    def name(self): 
        return str(self._name)
    
    @property
    def head(self):
        return self._head
    
    @property
    def tail(self):
        return self._tail
    
    @property
    def allen(self): # 基因總長
        return self._allen
    
    @property
    def pset(self): # 所使用符號集
        return self._pset
    
    @property
    def funcSet(self):
        return self._pset.functions
        
    @property
    def termSet(self):
        return self._pset.terminals
    
    @property
    def Dc(self):
        return self._dc

class GeneDc:
    def __init__(self, dcName, dcLen, rncLen, threshold):
        self._dcName = dcName
        self._dcLen = dcLen
        self._rncLen = rncLen
        self._threshold = threshold
        
    @property
    def allen(self):
        return self._dcLen + self._rncLen
    
    @property
    def dcLen(self):
        return self._dcLen
    
    @property
    def rncLen(self):
        return self._rncLen
    
    @property
    def threshold(self):
        return self._threshold


class TradeGene:
    def __init__(self, genome, genetype):
        self._genome = genome
        self._gentype = genetype
    
    @property
    def genome(self):
        return self._genome
    
    @property
    def head(self):
        return self._gentype.head
    
    @property
    def funcSet(self):
        return self._gentype.funcSet
    
    @property
    def termSet(self):
        return self._gentype.termSet
    
    @property
    def Dc(self): 
        return self._gentype.Dc
        
    @property
    def allen(self): # 基因總長
        return self._gentype.allen
    
class ZsorseGene:
    def __init__(self, genome, genetype):
        self._genome = genome
        self._gentype = genetype
    
    @property
    def genome(self):
        return self._genome
    
    @property
    def head(self):
        return self._gentype.head
    
    @property
    def funcSet(self):
        return self._gentype.funcSet
    
    @property
    def termSet(self):
        return self._gentype.termSet
    
    @property
    def Dc(self): 
        return self._gentype.Dc
        
    @property
    def allen(self): # 基因總長
        return self._gentype.allen
        
        
class FundMgtGene:
    def __init__(self, genome, gene):
        self._genome = genome
        
        
class RiskGene:
    def __init__(self, genome, genetype):
        self._genome = genome
        self._gentype = genetype
        
    @property
    def genome(self):
        return self._genome
    
    @property
    def head(self):
        return self._gentype.head
    
    @property
    def funcSet(self):
        return self._gentype.funcSet
    
    @property
    def termSet(self):
        return self._gentype.termSet
    
    @property
    def Dc(self): 
        return self._gentype.Dc
    
    @property
    def allen(self): # 基因總長
        return self._gentype.allen
