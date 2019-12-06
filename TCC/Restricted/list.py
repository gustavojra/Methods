import os
import sys
import numpy as np
from TCCSD import *

temp = """
    unit bohr
    0 1
    O
    H 1 {:<1.5f}
    H 1 {:<1.5f} 2 104.52
    symmetry c1
"""

set {
    BASIS         cc-pvdz
    SCF_TYPE      pk
    E_CONVERGENCE 8
    MAXITER       50
}

eccsd = []
eccsd_t = []
ecas22 = []
ecas44 = []
ecas66 = []
tcc22 = []
tcc44 = []
tcc66 = []

ratios = np.array(range(7,31,1))/10.0

ratios = ratios[::-1]

for r in ratios:
    Re = 1.809*r
    mol = geometry(temp.format(Re, Re))
    
#    eccsd.append(energy(0))
#    eccsd_t.append(energy(0))

    X = TCCSD(mol)

    X.TCCSD(active_space = [4,5])
    ecas22.append(X.Ecas)
    tcc22.append(X.Ecc)

    X.TCCSD(active_space = [3,4,5,6])
    ecas44.append(X.Ecas)
    tcc44.append(X.Ecc)

    X.TCCSD(active_space = [2,3,4,5,6,7])
    ecas66.append(X.Ecas)
    tcc66.append(X.Ecc)

fline = '\n{:<7}   {:<15}    {:<15}    {:<15}    {:<15}    {:<15}    {:<15}    {:<15}    {:<15}'
line = '{:<2.5f}   {:<5.10f}    {:<5.10f}    {:<5.10f}    {:<5.10f}    {:<5.10f}    {:<5.10f}    {:<5.10f}    {:<5.10f}'
out = fline.format('R/Re', 'CCSD', 'CCSD(T)', 'CAS(2,2)', 'TCCSD(2,2)', 'CAS(4,4)', 'TCCSD(4,4)', 'CAS(6,6)', 'TCCSD(6,6)')
out += '\n'

for i in range(len(ratios)):
    out += line.format(ratios[i],eccsd[i], eccsd_t[i], ecas22[i], tcc22[i], ecas44[i], tcc44[i], ecas66[i], tcc66[i])
    out += '\n'
print_out(out)
    

