class Chromosome:
    # genes => 基因種類
    def __init__(self, genes):
        self._genes = genes
      
    @property
    def genes(self):
        return self._genes 
     
    @property
    def genesName(self):
        return [i.geneName for i in self._genes]