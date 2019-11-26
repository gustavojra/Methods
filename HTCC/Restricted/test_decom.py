import os
import sys
import numpy as np
from CASDecom import CASDecom

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *
from ampcomp import tcompare
from CASCI import CASCI
from CASDecom import CASDecom
from fock import Det


ref = Det( a = '111000', b = '111000')

quad = Det( a = '000111', b = '110001')

d1 = Det( a = '110001', b = '110001')
d2 = Det( a = '011100', b = '011100')
d3 = Det( a = '011010', b = '101100')

dets = [ref, quad, d1, d2, d3]
C = np.array([1, 0.5, 0.6, 0.3, 0.1])

CASDecom

t1, t2, t3, t4abab, t4abaa = CASDecom(C, dets, ref)

p = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2, 0, 1], [2, 1, 0]]

print(t2)
print(t2 - t2.swapaxes(2,3))

for x in p:
    for y in p:
       [i,k,l] = x 
       [a,c,d] = y
       z = t4abaa[i,2,k,l,a,2,c,d]
       print(x)
       print(y)
       print(z)
