import psi4
import sys
import scipy.linalg as sp
sys.path.append('../Aux/')
from BCCD import CCSD
from tools import *

water = psi4.geometry("""
    O
    H 1 R
    H 1 R 2 A
    
    R = .9
    A = 104.5
    symmetry c1
""")

psi4.core.be_quiet()
psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence' : 1e-10,
                  'brueckner_orbs_r_convergence' : 1e-8,
                  'freeze_core': 'false'})

hf_e, wfn = psi4.energy('scf', return_wfn = True)

C = wfn.Ca().np

wfn_ccsd = CCSD(wfn) 

T1 = wfn_ccsd.T1amp['IA']
nvir = wfn_ccsd.avir
ndocc = wfn_ccsd.nalpha

ite = 1
while np.max(T1) > 1.e-8:
    if ite > 30:
        break
    X = np.block([
                   [ np.zeros([ndocc, ndocc]), np.zeros_like(T1)],
                   [ T1.T, np.zeros([nvir, nvir])]])
        
    U = sp.expm(X - X.T)
    C = C.dot(U)
    wfn_ccsd = CCSD(wfn, Cinp=psi4.core.Matrix.from_array(C))
    T1 = wfn_ccsd.T1amp['IA']

Ebru = wfn_ccsd.Ecc + wfn_ccsd.Escf
printmatrix(T1)
print(Ebru)
print(psi4.energy('bccd'))
