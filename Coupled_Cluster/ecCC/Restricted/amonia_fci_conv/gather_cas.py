import numpy as np
import re
import os

patt_cas = '@@@ Final CASCI\s+?energy:\s*?(.\d*?\.\d*)'
patt_tcc = '@@@ Final TCCSD\s+?energy:\s*?(.\d*?\.\d*)'
patt_cascc = '@@@ Final CASCCSD\s+?energy:\s*?(.\d*?\.\d*)'

c = re.compile(patt_cas)
y = re.compile(patt_tcc)
x = re.compile(patt_cascc)

acs = [(4,4), (6,6), (8,8), (10,10), (10,11), (10,12), (10,13), (10,14), (10,15)]

cas = []
tcc = []
cascc = []

for a in acs:
    dir_name = 'CAS_{}_{}'.format(a[0], a[1])
    os.chdir(dir_name)
    with open('output.dat') as out:
        out_str = out.read()
        try:
            ecas = c.search(out_str).group(1)
            print('CAS     ({},{}): {:<5.10f}'.format(a[0], a[1], float(ecas)))
            etcc = y.search(out_str).group(1)
            print('TCCSD   ({},{}): {:<5.10f}'.format(a[0], a[1], float(etcc)))
            ecascc = x.search(out_str).group(1)
            print('CASCCSD ({},{}): {:<5.10f}'.format(a[0], a[1], float(ecascc)))
            cas.append(float(ecas))
            tcc.append(float(etcc))
            cascc.append(float(ecascc))
        except:
            print('Problem with CAS ({},{})'.format(a[0],a[1]))
    os.chdir('..')

out = 'CAS SPACE   {:^15s}   {:^15s}   {:^15s}\n'.format('CAS', 'TCCSD', 'CASCCSD')

for a, c1, c2, c3 in zip(acs,cas,tcc,cascc):
    out += '({},{})   {:<5.10f}   {:<5.10f}   {:<5.10f}\n'.format(a[0], a[1], c1, c2, c3)

with open('energies.dat', 'w') as outp:
    outp.write(out)
        
