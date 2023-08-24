import os
import pandas as pd
import numpy as np
from gep2.stockdata import *
from gep2.tools import *
import matplotlib.pyplot as plt   
import matplotlib.dates as mdates
import mplfinance as mpf
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
import IPython.display as IPydisplay
from gep2.primitiveset import *
from gep2.multigene import TradeGene, RiskGene
import random
import datetime
import pandas as pd
import numpy as np
import copy
import mplfinance as mpf
'''
def get_is_pos(gene, head):
    geneLen = 0
    for node in gene:
        if isinstance(node, Node):
            geneLen += 1
    g_p1 = random.randint(1, geneLen) # IS插入起始
    g_p2 = random.randint(g_p1+1, geneLen) if g_p1 != geneLen else random.randint(1, g_p1-1) # IS插入終點
    h_p1 = random.randint(1, head-1) # 頭部插入起始
    if g_p1 > g_p2: g_p1, g_p2 = g_p2, g_p1
    return [g_p1, g_p2, h_p1]

def is_transpose(genome):
    genes = copy.deepcopy(genome)
    for gene in genes:
        head = gene.head
        allen = gene.allen
        if head > 1:
            pos = get_is_pos(gene.genome, head)
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
    return genes'''




rr=[0.015049142603656982, 0.013300218633730994, 0.018124897017630454, -0.5145721024258759, -0.19405760553920992, 0.14444444444444443, 0.04347826086956523, -0.012195121951219502, 0.08333333333333354, 0.34444444444444433, 0.18683530100497625, 0.05556823368324976, 0.029641667009503252, 0.31409856519026835, 0.2127139364303179, 0.18025751072961388, 0.07142857142857133, -0.12400000000000005, 0.8295218295218292, -0.38969764837625975, -0.028205543848273576, -0.31107406525363185, -0.29994425863991087, 0.2571831364124597, -0.023958736299161888, 0.07932960893854744, -0.21753246753246755, 0.554263565891473, 0.16571428571428568, 0.36206896551724127, 0.07111385947282986, 0.0640899661535103, -0.11937461592328474, 0.3978108984186782, 0.17681660063774277, -0.25619834710743816, 0.5415704387990761, -0.27902240325865585, 0.08628318584070789, 0.41223404255319146, 0.9514777542372878, 0.49304993252361695, -0.11176905995558839, 0.4223793597706641, 0.6146619326999507, -0.23724489795918366, 0.2896253602305477, 0.22357723577235789, 0.15912128983013035, 0.6494845360824746]
newrr = list()
for i in range(0, len(rr), 5):
    r = rr[i: i+5]
    newrr.append(sum(r)/len(r))

df = pd.DataFrame(newrr, columns=['平均報酬率'])
df['total'] = df['平均報酬率'].cumsum()
df.total.plot() 
#df.to_excel('lgbm.xlsx')

'''
{'2016-05-16': [-0.1324310899420135,
                -243788.0, 0.6, -1.0218252018122513, 261691.69802684974, 14.616328679020224, -0.9315847687876869, 0.068, 0.046, -8.095652173913045], '2016-11-15': [0.12070107222811362, 254060.0, 0.8, 1.0309778724005045, 6161.249209242895, 0.11821517509609325, 41.23514426569013, 42.235, 10.559, 0.7810588123875367], '2017-05-16': [0.15977154066366311, 385613.0, 1.0, 2.330770518685527, 0, 0, 385613.0, 385612.906, 77122.581, 1.0], '2017-11-15': [0.1135020526607509, 300430.0, 0.6, 0.47337036827341206, 183784.8410757946, 0.37955213273735866, 1.6346832428693063, 2.206, 1.471, 0.32807613868116925], '2018-05-16': [-0.0811998935257037, 58794.0, 0.2, 0.1860812578286855, 6576.5544202212695, 0.1006036821515781, 8.93993970751965, 1.475, 5.901, 0.06442975766819184], '2018-11-15': [0.18876879170581598, 561201.0, 0.8, 1.6196957186238983, 48334.82746531777, 0.6983239233818769, 11.61069625008354, 12.611, 3.153, 0.7365683476054552], '2019-05-16': [0.11809134175189528, 807317.0, 0.8, 1.1664807335330336, 73722.34170179308, 0.16721947080638122, 10.950778032331066, 11.951, 2.988, 0.7330655957161982], '2019-11-15': [0.10097338336537628, 969941.0, 0.6, 0.8837464780301151, 105651.78995346266, 0.15560815090147048, 9.18054488643532, 6.654, 4.436, 0.5098286744815148], '2020-05-16': [0.4739599838551863, 464076.0, 0.8, 0.37753384904812254, 599932.2804859667, 1.0737390162516964, 0.7735473070795287, 1.774, 0.443, 0.3485327313769753], '2020-11-15': [0.2169127047912654, 1028559.0, 0.8, 1.3879450807648088, 0, 0, 1028559.0, 7.905, 1.976, 0.6987854251012147]}


'''




