import sys
import re
import numpy as np

sys.path.append('../Modules/')

from Det import Det

def read_ci_vec(ref, fdocc, nmo):
    pattern = '\s*?\*\s*?\d+?\s+?([-\s]\d\.\d+?)\s+?\(.+?\)\s+?(.+?\n)'
    Ccas = []
    determinants = []
    dets_string = []
    with open('output.dat', 'r') as output:
        for line in output:
            m = re.match(pattern, line)
            if m:
                Ccas.append(float(m.group(1)))
                dets_string.append(m.group(2))
    
    for det in dets_string:
        a_index = []
        b_index = []
        for o in det.split():
            if o[-1] == 'X' or o[-1] == 'A':
                a_index.append(int(o[:-2])-1)
            if o[-1] == 'X' or o[-1] == 'B':
                b_index.append(int(o[:-2])-1)
        a_string = '1'*fdocc
        b_string = '1'*fdocc
        for i in range(fdocc,nmo):
            if i in a_index:
                a_string += '1'
            else:
                a_string += '0'
            if i in b_index:
                b_string += '1'
            else:
                b_string += '0'
        determinants.append(Det(a = a_string, b = b_string, ref = ref, sq = True))

    for i,d in enumerate(determinants):
        Ccas[i] *= d.order/abs(d.order)

    return Ccas, determinants
