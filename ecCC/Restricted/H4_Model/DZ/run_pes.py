import numpy as np
import sys
import os

def create_input(temp, alpha, a):
    nelec = 4
    active_elec = a[0]
    nfdocc = (nelec - active_elec)/2
    nact = a[1]
    angle = float(alpha*180 + 90)
    final = temp.format(angle, angle, frozen=nfdocc, active=nact, alpha=alpha) + temp_end
    with open('input.dat','w') as inp:
        inp.write(final)

# Vulcan submit
prefix = ". /opt/vulcan/share/vulcan/setup-env.sh; "
submit = "vulcan submit gen4.q,gen6.q psi4@master"

temp = \
"""import sys
sys.path.append('../../../Modules/')
from TCCSD import TCCSD
from CASCCSD import CASCCSD

molecule mol {{
    unit bohr
    0 1
    H
    H 1 2.0 
    H 1 2.0 2 {:<1.5} 
    H 2 2.0 1 {:<1.5} 3 0.0
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
    NUM_DETS_PRINT 220000
    FCI True
}}

set FROZEN_DOCC = [{frozen}]
set ACTIVE = [{active}]

e, wfn = energy('detci', return_wfn=True)
Y = TCCSD(wfn, CC_MAXITER=300)
X = CASCCSD(wfn, CC_MAXITER=300)

print_out('@@@ alpha = {alpha}')
"""

temp_end = \
r"""print_out('\n@@@ Final CASCI   energy: {:<5.10f}\n'.format(e))
print_out('@@@ Final TCCSD   energy: {:<5.10f}\n'.format(Y.Ecc))
print_out('@@@ Final CASCCSD   energy: {:<5.10f}\n'.format(X.Ecc))"""


# PES range and level of theory
#acs = [(2,2), (4,4), (4,8), (4,10), (4,12), (4,20)]
acs = [(4,6), (4,14), (4,16), (4,18)]
R = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for a in acs:
    dir_name = 'PES_{}_{}'.format(a[0], a[1])
    if os.path.exists(dir_name):
        continue
    os.makedirs(dir_name)
    os.chdir(dir_name)
    for i,r in enumerate(R):
        os.makedirs('p' + str(i))
        os.chdir('p'+str(i))
        create_input(temp, r, a)
        os.system(prefix + submit)
        os.chdir('..')
    os.chdir('..')
