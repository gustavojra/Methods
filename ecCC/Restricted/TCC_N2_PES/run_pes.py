import numpy as np
import sys
import os

def create_input(temp, r, a):
    nelec = 14
    active_elec = a[0]
    nfdocc = (nelec - active_elec)/2
    nact = a[1]
    final = temp.format(r, frozen=nfdocc, active=nact, ratio=r/2.074) + temp_end
    with open('input.dat','w') as inp:
        inp.write(final)

# Vulcan submit
prefix = ". /opt/vulcan/share/vulcan/setup-env.sh; "
submit = "vulcan submit gen4.q,gen6.q psi4@master"

temp = \
"""import sys
sys.path.append('../../../Modules/')
from TCCSD import TCCSD

molecule mol {{
    unit bohr
    0 1
    N 
    N 1 {:1.5f}
    symmetry c1
}}

set {{
    BASIS         cc-pvdz
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
X = TCCSD(wfn, CC_MAXITER=300)

print_out('@@@ R/Re = {ratio}')
"""

temp_end = \
r"""print_out('\n@@@ Final CASCI   energy: {:<5.10f}\n'.format(e))
print_out('@@@ Final TCCSD   energy: {:<5.10f}\n'.format(X.Ecc))"""


# PES range and level of theory
acs = [(4,4), (6,6), (8,7), (10,8), (10,9), (10,11), (10,12), (14,10)]
R = np.array(range(6,32,2))/10

for a in acs:
    dir_name = 'PES_{}_{}'.format(a[0], a[1])
    if os.path.exists(dir_name):
        continue
    os.makedirs(dir_name)
    os.chdir(dir_name)
    for i,r in enumerate(R):
        os.makedirs('p' + str(i))
        os.chdir('p'+str(i))
        create_input(temp, r*2.074, a)
        os.system(prefix + submit)
        os.chdir('..')
    os.chdir('..')
