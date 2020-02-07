import numpy as np
import sys
import os

def create_input(temp, a):
    e = 10
    ae = a[0]
    nfdocc = (e - ae)/2
    ao = a[1]
    final = temp.format(frozen=nfdocc, active=ao, e=ae, o=ao) + temp_end
    with open('input.dat','w') as inp:
        inp.write(final)

# Vulcan submit
prefix = ". /opt/vulcan/share/vulcan/setup-env.sh; "
submit = "vulcan submit gen4.q,gen6.q psi4@master"

temp = \
"""import sys
sys.path.append('../../Modules/')
from TCCSD import TCCSD
from CASCCSD import CASCCSD

molecule mol {{
    0 1
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1
}}

set {{
    BASIS         6-31g
    REFERENCE     RHF
    SCF_TYPE      PK
    SOSCF         True
    E_CONVERGENCE 12
    MAXITER       100
    MAX_ATTEMPTS  10
    NUM_DETS_PRINT 100000
    FCI True
}}

set FROZEN_DOCC = [{frozen}]
set ACTIVE = [{active}]

e, wfn = energy('detci', return_wfn=True)
Y = TCCSD(wfn, CC_MAXITER=300)
X = CASCCSD(wfn, CC_MAXITER=300)

print_out('@@@ CAS = ({e},{o})')
"""

temp_end = \
r"""print_out('\n@@@ Final CASCI   energy: {:<5.10f}\n'.format(e))
print_out('@@@ Final TCCSD   energy: {:<5.10f}\n'.format(Y.Ecc))
print_out('@@@ Final CASCCSD   energy: {:<5.10f}\n'.format(X.Ecc))"""


# PES range and level of theory
acs = [(4,4), (6,6), (8,8), (10,10), (10,11), (10,12), (10,13)]

for a in acs:
    dir_name = 'CAS_{}_{}'.format(a[0], a[1])
    if os.path.exists(dir_name):
        continue
    os.makedirs(dir_name)
    os.chdir(dir_name)
    create_input(temp, a)
    os.system(prefix + submit)
    os.chdir('..')
