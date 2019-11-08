import psi4
import os
import sys
import numpy as np
import time

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *

### FUNCTIONS ###

def cc_energy(T2):
    return (1/4)*np.einsum('ijab,ijab->',A[o,o,v,v],T2)
    #return (1/4)*np.sum(A[o,o,v,v]*T2)

def T2_iter(T2):

    w1 = np.einsum('kbcj,ikac->ijab', A[o,v,v,o], T2)
    w2 = 1.0/2.0 * np.einsum('klcd,ikac,ljdb->ijab', A[o,o,v,v], T2, T2)
    w3 = -1.0/2.0 * np.einsum('klcd,ijac,klbd->ijab', A[o,o,v,v], T2, T2)
    w4 = -1.0/2.0 * np.einsum('klcd,ikab,jlcd->ijab', A[o,o,v,v], T2, T2)

    T2new = A[o,o,v,v]                                                                               \
    + 1.0/2.0 * np.einsum('klij,klab->ijab', A[o, o, o, o], T2)                                      \
    + 1.0/2.0 * np.einsum('abcd,ijcd->ijab', A[v, v, v, v], T2)                                      \
    + w1 - w1.transpose(1,0,2,3) - w1.transpose(0,1,3,2) + w1.transpose(1,0,3,2)                     \
    + w2.transpose(0,1,2,3) - w2.transpose(1,0,2,3) - w2.transpose(0,1,3,2) + w2.transpose(1,0,3,2)  \
    + 1.0/4.0 * np.einsum('klcd,ijcd,klab->ijab', A[o,o,v,v], T2, T2)                                \
    + w3.transpose(0,1,2,3) - w3.transpose(0,1,3,2)                                                  \
    + w4.transpose(0,1,2,3) - w4.transpose(1,0,2,3)                                                  

    T2new = np.einsum('ijab,ijab->ijab', T2new, D)

    res = np.sum(np.abs(T2new - T2))

    return T2new, res


def A_T2_iter(t):
   g = A
   w1 = +1.   * np.einsum("akic,jkbc->ijab", g[v,o,o,v], t)
   w2 = -1./2 * np.einsum("klcd,ijac,klbd->ijab", g[o,o,v,v], t, t)
   w3 = -1./2 * np.einsum("klcd,ikab,jlcd->ijab", g[o,o,v,v], t, t)
   w4 = +1.   * np.einsum("klcd,ikac,jlbd->ijab", g[o,o,v,v], t, t)
      # update T2 amplitudes
   T2new  = g[o,o,v,v]                                             \
      + 1./2 * np.einsum("abcd,ijcd->ijab", g[v,v,v,v], t)         \
      + 1./2 * np.einsum("klij,klab->ijab", g[o,o,o,o], t)         \
      + w1.transpose((0,1,2,3)) - w1.transpose((0,1,3,2))          \
      - w1.transpose((1,0,2,3)) + w1.transpose((1,0,3,2))          \
      + w2.transpose((0,1,2,3)) - w2.transpose((0,1,3,2))          \
      + w3.transpose((0,1,2,3)) - w3.transpose((1,0,2,3))          \
      + 1./4 * np.einsum("klcd,ijcd,klab->ijab", g[o,o,v,v], t, t) \
      + w4.transpose((0,1,2,3)) - w4.transpose((1,0,2,3))
   T2new *= D

   res = np.sum(np.abs(T2new - t))

   return T2new, res

# Input Geometry    

#H2 = psi4.geometry("""
#    0 1
#    H 
#    H 1 0.76
#    symmetry c1
#""")

water = psi4.geometry("""
    0 1
    O
    H 1 0.96
    H 1 0.96 2 104.5
    symmetry c1
""")
#
#form = psi4.geometry("""
#0 1
#O
#C 1 1.22
#H 2 1.08 1 120.0
#H 2 1.08 1 120.0 3 -180.0
#symmetry c1
#""")

# Basis set

basis = 'sto-3g'

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': basis,
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence' : 1e-10,
                  'freeze_core': 'false'})

# Run Psi4 Energie
print('---------------- RUNNING PSI4 ------------------')
t = time.time()
scf_e, wfn = psi4.energy('scf', return_wfn=True)
p4_mp2 = psi4.energy('mp2')
p4_ccsd = psi4.energy('ccsd')

print('SCF  Energy from Psi4: {:<5.10f}'.format(scf_e))
print('MP2  Energy from Psi4: {:<5.10f}'.format(p4_mp2))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print('------------------------------------------------')
print('Psi4 computations completed in {:.5f} seconds\n'.format(time.time() - t))

nelec = wfn.nalpha() + wfn.nbeta()
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nocc = ndocc * 2 + nelec % 2
nmo = wfn.nmo()
nso = nmo * 2
nvir = nso - nocc
eps = np.asarray(wfn.epsilon_a())
eps = np.repeat(eps, 2)
nbf = C.shape[0]

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Electrons:            {}".format(nelec))
print("Number of Spin-Orbitals (SO):   {}".format(nso))
print("Number of Ocuppied SO:          {}".format(nocc))

# Get Integrals

mints = psi4.core.MintsHelper(wfn.basisset())
V = np.asarray(mints.mo_spin_eri(C,C)).swapaxes(1,2)

# Slices

o = slice(0, nocc)
v = slice(nocc, 2*nbf)

# START CCD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrix...')
t = time.time()
D  = np.zeros([nocc, nocc, nvir, nvir])

for i,ei in enumerate(eps[o]):
    for j,ej in enumerate(eps[o]):
        for a,ea in enumerate(eps[v]):
            for b,eb in enumerate(eps[v]):
                D[i,j,a,b] = 1/(ei + ej - ea - eb)
print('Done. Time required: {:.5f} seconds'.format(time.time() - t))

#Aoovv = V[o,o,v,v] - np.einsum('ijab->ijba',V[o,o,v,v])

A = V - np.einsum('ijab->ijba',V)

print('\nComputing MP2 guess')

t = time.time()

T2 = np.einsum('ijab,ijab->ijab', A[o,o,v,v], D)

Eold = cc_energy(T2)

print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(Eold+scf_e, time.time()-t))

r2 = 1
CC_CONV = 6
CC_MAXITER = 50
    
LIM = 10**(-CC_CONV)

ite = 0

while r2 > LIM:
    ite += 1
    if ite > CC_MAXITER:
        raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
    t = time.time()
    T2,r2 = T2_iter(T2)
    E = cc_energy(T2)
    print('-'*50)
    print("Iteration {}".format(ite))
    print("CC Correlation energy: {}".format(E))
    print("Total energy:          {}".format(E+scf_e))
    print("T2 Residue:            {}".format(r2))
    print("Max Amplitude:         {}".format(np.max(T2)))
    print("Time required:         {}".format(time.time() - t))
    print('-'*50)

print("\nCC Equations Converged!!!")
print("Final CCD Energy: {}".format(E + scf_e))


