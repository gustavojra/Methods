import numpy as np
import re
import os

patt_cas = '@@@ Final CASCI\s+?energy:\s*?(.\d*?\.\d*)'
patt_tcc = '@@@ Final TCCSD\s+?energy:\s*?(.\d*?\.\d*)'
patt_cascc = '@@@ Final CASCCSD\s+?energy:\s*?(.\d*?\.\d*)'

c = re.compile(patt_cas)
y = re.compile(patt_tcc)
x = re.compile(patt_cascc)

acs = [(2,2), (4,4), (4,8), (4,10), (4,12), (4,20)]

R = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


for a in acs:
    cas = []
    tcc = []
    cascc = []
    dir_name = 'PES_{}_{}'.format(a[0], a[1])
    os.chdir(dir_name)
    for i in range(len(R)):
        point = 'p'+str(i)
        os.chdir(point)
        with open('output.dat') as out:
            out_str = out.read()
            try:
                ecas = c.search(out_str).group(1)
                etcc = y.search(out_str).group(1)
                ecascc = x.search(out_str).group(1)
                cas.append(float(ecas))
                tcc.append(float(etcc))
                cascc.append(float(ecascc))
            except:
                print('Problem with CAS ({},{}). Point {}'.format(a[0],a[1], i))
        os.chdir('..')

    out = 'Alpha   {:^15s}   {:^15s}   {:^15s}\n'.format('CAS', 'TCCSD', 'CASCCSD')
    
    for a, c1, c2, c3 in zip(R,cas,tcc,cascc):
        out += '{:1.2f}   {:<5.10f}   {:<5.10f}   {:<5.10f}\n'.format(a, c1, c2, c3)
    
    with open('energies.dat', 'w') as outp:
        outp.write(out)
    os.chdir('..')
        
