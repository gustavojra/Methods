import psi4
import os
import sys
import numpy as np
import time
import copy

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *

### FUNCTIONS ###

def cc_energy(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab):

    X = T2aaaa + 2*np.einsum('ia,jb->ijab',T1aa,T1aa, optimize='optimal')
    E = np.einsum('ijab,ijab->', X, Vaaaa[oa,oa,va,va], optimize='optimal')

    X = T2abab + 2*np.einsum('ia,jb->ijab',T1aa,T1bb, optimize='optimal')
    E += np.einsum('ijab,ijab->', X, Vabab[oa,ob,va,vb], optimize='optimal')

    E += np.einsum('ijab,ijab->', T2abba, Vabba[oa,ob,vb,va], optimize='optimal')

    X = T2bbbb + 2*np.einsum('ia,jb->ijab',T1bb,T1bb, optimize='optimal')
    E += np.einsum('ijab,ijab->', X, Vbbbb[ob,ob,vb,vb], optimize='optimal')

    X = T2baba + 2*np.einsum('ia,jb->ijab',T1bb,T1aa, optimize='optimal')
    E += np.einsum('ijab,ijab->', X, Vbaba[ob,oa,vb,va], optimize='optimal')

    E += np.einsum('ijab,ijab->', T2baab, Vbaab[ob,oa,va,vb], optimize='optimal')

    return (1.0/4.0) * E

def CCSD_Iter(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)

    # F(ae) Intermediates

    ## F(ae) (alpha|alpha)

    Fae_aa = np.einsum('mf,amef->ae', T1aa, Vaaaa[va,oa,va,va], optimize='optimal') + np.einsum('mf,amef->ae', T1bb, Vabab[va,ob,va,vb], optimize='optimal')

    X = T2aaaa + 0.5 *(np.einsum('ma,nf->mnaf', T1aa, T1aa, optimize='optimal') - np.einsum('mf,na->mnaf', T1aa, T1aa, optimize='optimal'))

    Fae_aa += -0.5 * np.einsum('mnaf, mnef -> ae', X, Vaaaa[oa,oa,va,va], optimize='optimal')

    X = T2abab + 0.5 *np.einsum('ma,nf->mnaf', T1aa, T1bb, optimize='optimal') 

    Fae_aa += -0.5 * np.einsum('mnaf, mnef -> ae', X, Vabab[oa,ob,va,vb], optimize='optimal')

    X = T2baab - 0.5 * np.einsum('mf,na->mnaf', T1bb, T1aa, optimize='optimal')

    Fae_aa += -0.5 * np.einsum('mnaf, mnef -> ae', X, Vbaab[ob,oa,va,vb], optimize='optimal')

    ## F(ae) (beta|beta)

    Fae_bb = np.einsum('mf,amef->ae', T1bb, Vbbbb[vb,ob,vb,vb], optimize='optimal') + np.einsum('mf,amef->ae', T1bb, Vbaba[vb,oa,vb,va], optimize='optimal')

    X = T2bbbb + 0.5 *(np.einsum('ma,nf->mnaf', T1bb, T1bb, optimize='optimal') - np.einsum('mf,na->mnaf', T1bb, T1bb, optimize='optimal'))

    Fae_bb += -0.5 * np.einsum('mnaf, mnef -> ae', X, Vbbbb[ob,ob,vb,vb], optimize='optimal')

    X = T2baba + 0.5 *np.einsum('ma,nf->mnaf', T1bb, T1aa, optimize='optimal') 

    Fae_bb += -0.5 * np.einsum('mnaf, mnef -> ae', X, Vbaba[ob,oa,vb,va], optimize='optimal')

    X = T2abba - 0.5 * np.einsum('mf,na->mnaf', T1aa, T1bb, optimize='optimal')

    Fae_bb += -0.5 * np.einsum('mnaf, mnef -> ae', X, Vabba[oa,ob,vb,va], optimize='optimal')

    # F(mi) Intermediates

    ## F(mi) (alpha|alpha)

    Fmi_aa = np.einsum('ne,mnie->mi', T1aa, Vaaaa[oa,oa,oa,va], optimize='optimal') + np.einsum('ne,mnie->mi', T1bb, Vabab[oa,ob,oa,vb], optimize='optimal')

    X = T2aaaa + 0.5 *(np.einsum('ie,nf->inef', T1aa, T1aa, optimize='optimal') - np.einsum('if,ne->inef', T1aa, T1aa, optimize='optimal'))

    Fmi_aa += +0.5 * np.einsum('inef, mnef -> mi', X, Vaaaa[oa,oa,va,va], optimize='optimal')

    X = T2abab + 0.5 *np.einsum('ie,nf->inef', T1aa, T1bb, optimize='optimal') 

    Fmi_aa += +0.5 * np.einsum('inef, mnef -> mi', X, Vabab[oa,ob,va,vb], optimize='optimal')

    X = T2abba - 0.5 * np.einsum('if,ne -> inef', T1aa, T1bb, optimize='optimal')

    Fmi_aa += +0.5 * np.einsum('inef, mnef -> mi', X, Vabba[oa,ob,vb,va], optimize='optimal')

    ## F(mi) (beta|beta)

    Fmi_bb = np.einsum('ne,mnie->mi', T1bb, Vbbbb[ob,ob,ob,vb], optimize='optimal') + np.einsum('ne,mnie->mi', T1aa, Vbaba[ob,oa,ob,va], optimize='optimal')

    X = T2bbbb + 0.5 *(np.einsum('ie,nf->inef', T1bb, T1bb, optimize='optimal') - np.einsum('if,ne->inef', T1bb, T1bb, optimize='optimal'))

    Fmi_bb += +0.5 * np.einsum('inef, mnef -> mi', X, Vbbbb[ob,ob,vb,vb], optimize='optimal')

    X = T2baba + 0.5 *np.einsum('ie,nf->inef', T1bb, T1aa, optimize='optimal') 

    Fmi_bb += +0.5 * np.einsum('inef, mnef -> mi', X, Vbaba[ob,oa,vb,va], optimize='optimal')

    X = T2baab - 0.5 * np.einsum('if,ne -> inef', T1bb, T1aa, optimize='optimal')

    Fmi_bb += +0.5 * np.einsum('inef, mnef -> mi', X, Vbaab[ob,oa,va,vb], optimize='optimal')

    # F(me) Intermediates

    ## F(me) (alpha|alpha)

    Fme_aa = np.einsum('nf,mnef->me', T1aa, Vaaaa[oa,oa,va,va], optimize='optimal') + np.einsum('nf,mnef->me', T1bb, Vabab[oa,ob,va,vb], optimize='optimal')

    ## F(me) (beta|beta)

    Fme_bb = np.einsum('nf,mnef->me', T1bb, Vbbbb[ob,ob,vb,vb], optimize='optimal') + np.einsum('nf,mnef->me', T1aa, Vbaba[ob,oa,vb,va], optimize='optimal')

    # W (mnij) Intermediates

    ## W (mnij) (alpha|alpha|alpha|alpha)

    Wmnij_aaaa = V




# Input Geometry    

water = psi4.geometry("""
    0 2
    O
    H 1 0.96
    symmetry c1
""")

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': 'sto-3g',
                  'reference': 'uhf',                
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence' : 1e-10,
                  'freeze_core': 'false'})

# Run Psi4 Energie
print('---------------- RUNNING PSI4 ------------------')
tinit = time.time()
scf_e, wfn = psi4.energy('scf', return_wfn=True)
p4_mp2 = psi4.energy('mp2')
p4_ccsd = psi4.energy('ccsd')

print('SCF  Energy from Psi4: {:<5.10f}'.format(scf_e))
print('MP2  Energy from Psi4: {:<5.10f}'.format(p4_mp2))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print('------------------------------------------------')
print('Psi4 computations completed in {:.5f} seconds\n'.format(time.time() - tinit))

neleca = wfn.nalpha()
nelecb = wfn.nbeta()
nelec = neleca + nelecb
Ca = wfn.Ca()
Cb = wfn.Cb()
ndocc = wfn.doccpi()[0] # Is this meaningfull?
nbf = Ca.shape[0]
naso = Ca.shape[0] 
nbso = Cb.shape[0]
navir = naso - neleca
nbvir = nbso - nelecb
epsa = np.asarray(wfn.epsilon_a())
epsb = np.asarray(wfn.epsilon_b())

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Alpha Electrons:      {}".format(neleca))
print("Number of Beta Electrons:       {}".format(nelecb))
print("Number of Alpha Spin-Orbitals:  {}".format(naso))
print("Number of Beta Spin-Orbitals:   {}".format(nbso))

# Get Integrals

print("Creating Antisymmetrized MO Integrals")

t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())

print("AA|AA")
Vaaaa = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))
Vaaaa = Vaaaa - Vaaaa.transpose(0,3,2,1)
# Physicist's Notation
Vaaaa = Vaaaa.transpose(0,2,1,3)

print("BB|BB")
Vbbbb = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))
Vbbbb = Vbbbb - Vbbbb.transpose(0,3,2,1)
# Physicist's Notation
Vbbbb = Vbbbb.transpose(0,2,1,3)

print("AB|AB")
Vabab = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))
# Physicist's Notation
Vabab = Vabab.transpose(0,2,1,3)

print("AB|BA")
Vabba = (-1)*np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))
# Physicist's Notation
Vabba = Vabba.transpose(0,2,3,1)

print("BA|BA")
Vbaba = np.asarray(mints.mo_eri(Cb, Cb, Ca, Ca))
# Physicist's Notation
Vbaba = Vbaba.transpose(0,2,1,3)

print("BA|AB")
Vbaab = (-1)*np.asarray(mints.mo_eri(Cb, Cb, Ca, Ca))
# Physicist's Notation
Vbaab = Vbaab.transpose(0,2,3,1)

print("Completed in {} seconds!".format(time.time()-t))

# Slices

oa = slice(0, neleca)
va = slice(neleca, naso)

ob = slice(0, nelecb)
vb = slice(nelecb, nbso)

# START CCSD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrices...')
t = time.time()

print('d(AA)')
daa  = np.zeros([neleca, navir])
for i,ei in enumerate(epsa[oa]):
    for a,ea in enumerate(epsa[va]): 
        daa[i,a] = 1/(ei - ea)

print('d(BB)')
dbb  = np.zeros([nelecb, nbvir])
for i,ei in enumerate(epsb[ob]):
    for a,ea in enumerate(epsb[vb]): 
        dbb[i,a] = 1/(ei - ea)

print('D(AAAA)')
Daaaa  = np.zeros([neleca, neleca, navir, navir])
for i,ei in enumerate(epsa[oa]):
    for j,ej in enumerate(epsa[oa]):
        for a,ea in enumerate(epsa[va]):
            for b,eb in enumerate(epsa[va]):
                Daaaa[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(BBBB)')
Dbbbb  = np.zeros([nelecb, nelecb, nbvir, nbvir])
for i,ei in enumerate(epsb[ob]):
    for j,ej in enumerate(epsb[ob]):
        for a,ea in enumerate(epsb[vb]):
            for b,eb in enumerate(epsb[vb]):
                Dbbbb[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(ABAB)')
Dabab  = np.zeros([neleca, nelecb, navir, nbvir])
for i,ei in enumerate(epsa[oa]):
    for j,ej in enumerate(epsb[ob]):
        for a,ea in enumerate(epsa[va]):
            for b,eb in enumerate(epsb[vb]):
                Dabab[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(ABBA)')
Dabba  = np.zeros([neleca, nelecb, nbvir, navir])
for i,ei in enumerate(epsa[oa]):
    for j,ej in enumerate(epsb[ob]):
        for a,ea in enumerate(epsb[vb]):
            for b,eb in enumerate(epsa[va]):
                Dabba[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(BABA)')
Dbaba  = np.zeros([nelecb, neleca, nbvir, navir])
for i,ei in enumerate(epsb[ob]):
    for j,ej in enumerate(epsa[oa]):
        for a,ea in enumerate(epsb[vb]):
            for b,eb in enumerate(epsa[va]):
                Dbaba[i,j,a,b] = 1/(ei + ej - ea - eb)

print('D(BAAB)')
Dbaab  = np.zeros([nelecb, neleca, navir, nbvir])
for i,ei in enumerate(epsb[ob]):
    for j,ej in enumerate(epsa[oa]):
        for a,ea in enumerate(epsa[va]):
            for b,eb in enumerate(epsb[vb]):
                Dbaab[i,j,a,b] = 1/(ei + ej - ea - eb)

print('Done. Time required: {:.5f} seconds'.format(time.time() - t))

print('\nComputing MP2 guess')

t = time.time()

T1aa = np.zeros([neleca, navir])
T1bb = np.zeros([nelecb, nbvir])

T2aaaa = np.einsum('ijab,ijab->ijab', Vaaaa[oa,oa,va,va], Daaaa)
T2bbbb = np.einsum('ijab,ijab->ijab', Vbbbb[ob,ob,vb,vb], Dbbbb)
T2abab = np.einsum('ijab,ijab->ijab', Vabab[oa,ob,va,vb], Dabab)
T2abba = np.einsum('ijab,ijab->ijab', Vabba[oa,ob,vb,va], Dabba)
T2baba = np.einsum('ijab,ijab->ijab', Vbaba[ob,oa,vb,va], Dbaba)
T2baab = np.einsum('ijab,ijab->ijab', Vbaab[ob,oa,va,vb], Dbaab)

E = cc_energy(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)

print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(E+scf_e, time.time()-t))

r1 = 0
r2 = 1
CC_CONV = 6
CC_MAXITER = 30
    
LIM = 10**(-CC_CONV)

ite = 0

while r2 > LIM or r1 > LIM:
    ite += 1
    if ite > CC_MAXITER:
        raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
    Eold = E
    t = time.time()

    # Update Amplitudes
    T1aa, T1bb, T2aaaa, T2abab, Tabba, T2bbbb, T2baba, T2baab, r1, r2  \
        = CCSD_Iter(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)

    # Update Energy
    E = cc_energy(T1aa, T1bb, T2aaaa, T2abab, T2abba, T2bbbb, T2baba, T2baab)
    dE = E - Eold

    print('-'*50)
    print("Iteration {}".format(ite))
    print("CC Correlation energy: {}".format(E))
    print("Energy change:         {}".format(dE))
    print("T1 Residue:            {}".format(r1))
    print("T2 Residue:            {}".format(r2))
    print("Time required:         {}".format(time.time() - t))
    print('-'*50)

print("\nCC Equations Converged!!!")
print("Final CCSD Energy:     {:<5.10f}".format(E + scf_e))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print("Total Computation time:        {}".format(time.time() - tinit))


