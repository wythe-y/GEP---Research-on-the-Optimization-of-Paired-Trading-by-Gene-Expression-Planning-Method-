from gep2.node import *

class RncForFunc:
    def __init__(self, threshold):
        self._min = threshold[0]
        self._max = threshold[1]
    
class RncForTerm:
    def __init__(self, threshold):
        self._min = threshold[0]
        self._max = threshold[1]

class PrimitiveSet:
    def __init__(self, name):
        self._name = name # primitive set
        self._functions = [] # array of function symbol
        self._terminals = [] # array of terminal symbol
        self._rnc_ary = [] # array of rnc array

    def add_function(self, functions, arity):
        for function in functions:
            function_ = FuncNode(function, arity)
            self._functions.append(function_)
        
    def add_terminal(self, terminals, arity=0): # terminal node no arity
        for terminal in terminals:
            terminal_ = TermNode(terminal, arity)
            self._terminals.append(terminal_)
           
    def add_rnc_terminal(self, name='?', arity=0):
        terminal_ = TermNode(name, arity)
        self._terminals.append(terminal_)
        
    def add_rnc_threshold(self, ary):
        for i in ary:
            self._rnc_ary.append(i)
    
    @property  
    def functions(self):
        return self._functions
    
    @property  
    def terminals(self):
        return self._terminals
    
    @property  
    def info(self):
        return '函式符號:{}  終端符號:{}'.format([i.name for i in self._functions], [i.name for i in self._terminals])
    
    @property
    def name(self):
        return self._name
    
    @property
    def max_arity(self):
        max_arity = 0
        if len(self.functions) > 0:
            max_arity = max(f.arity for f in self.functions)
        return max_arity