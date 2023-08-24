class Node:
    def __init__(self, name, arity):
        self._name = name 
        self._arity = arity
        self._child = None 
        self._period = None 
        self._minval = None
        self._maxval= None
        self._lot = None 
        self._risk = None
        self._minval = None 
        self._maxval = None
        self._lot = None
        
    def update_child(self, child):
        setattr(self, '_child', child)    
    
    def update_period(self, period):
        setattr(self, '_period', period)
        
    def update_lot(self, lot):
        setattr(self, '_lot', lot)
    
    def update_risk(self, risk):
        setattr(self, '_risk', risk)
        
    def update_arity(self, arity):
        setattr(self, '_arity', arity)
    
    def update_threshsold(self, min_, max_):
        setattr(self, '_minval', min_)
        setattr(self, '_maxval', max_)
        
    @property
    def name(self):
        return self._name
    
    @property
    def arity(self):
        return self._arity
    
    @property
    def period(self):
        return self._period
    
    @property
    def minval(self):
        return self._minval
    
    @property
    def maxval(self):
        return self._maxval
    
    @property
    def lot(self):
        return self._lot
    
    @property
    def risk(self):
        return self._risk
    
    @property
    def child(self):
        return self._child
    
class FuncNode:
    def __init__(self, name, arity):
        self._name = name
        self._arity = arity
        
    @property
    def name(self):
        return self._name
    
    @property
    def arity(self):
        return self._arity
    
class TermNode:
    def __init__(self, name, arity):
        self._name = name
        self._arity = arity
        
    @property
    def name(self):
        return self._name
    
    @property
    def arity(self):
        return self._arity
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        