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

def cc_energy(T1, T2):

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    X = 2*tau - np.einsum('ijab->jiab',tau)
    E = np.einsum('abij,ijab->', Vint[v,v,o,o], X)
    return E

def T2_iter(T1, T2):

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    V = 2*Vint - np.einsum('abij->baij', Vint)
    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1)
    
    T2new = copy.deepcopy(Vint[o,o,v,v])
    
    T2new += np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau)
    
    T2new += np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau)
    
    T2new += np.einsum('abij,ijpg,uvab->uvpg', Vint[v,v,o,o], tau, tau)
    
    T2new = 0.5*T2new
    
    T2new += np.einsum('uapg,va->uvpg', Vint[o,v,v,v], T1)
    
    T2new -= np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1)
    
    T2new += np.einsum('auip,viga->uvpg', V[v,o,o,v], T2)
    
    T2new -= np.einsum('uaig,ivpa->uvpg', Vint[o,v,o,v], tau) 
    
    T2new -= np.einsum('auip,viag->uvpg', Vint[v,o,o,v], tau)
    
    X = T2 - np.einsum('uipa->uiap',tau)
    
    T2new += np.einsum('abij,vjgb,uipa->uvpg', V[v,v,o,o], T2, X)
    
    T2new -= np.einsum('abij,ijgb,uvpa->uvpg', V[v,v,o,o], tau, T2)
    
    T2new -= np.einsum('abij,vjab,uipg->uvpg', V[v,v,o,o], tau, T2)
    
    T2new += np.einsum('abij,vjbg,uiap->uvpg', Vint[v,v,o,o], T2, Te)
    
    T2new += np.einsum('abij,ujag,vibp->uvpg', Vint[v,v,o,o], T2, Te)
    
    X = np.einsum('ivpg,ja->ivjpga', T2, T1) + np.einsum('jvag,ip->ivjpga', T2, T1)
    
    Y = np.einsum('vjag,ip->vjiagp', T2, T1) + np.einsum('viap,jg->vjiagp', T2, T1) + np.einsum('ijpg,va->vjiagp', tau, T1)
    
    T2new -= np.einsum('uaij,ivjpga,uaij,vjiagp->uvpg', V[o,v,o,o], X, Vint[o,v,o,o], Y)
    
    X = np.einsum('uvag,ib->uviagb', T2, T1) + np.einsum('ivbg,ua->uviagb', T2, T1)
    
    T2new += np.einsum('abpi,uviagb->uvpg', V[v,v,v,o], X)
    
    Y = np.einsum('ivgb,ua->ivugba', T2, T1) + np.einsum('iuga,vb->ivugba', T2, T1) + np.einsum('uvab,ig->ivugba', tau, T1)
    
    T2new -= np.einsum('abpi,ivugba->uvpg', Vint[v,v,v,o], Y)
    
    # Finish it up: Apply the permutation operator and divide by orbital energies [D(ijab)]
    
    # DEBUG
    
    #T2new -= np.einsum('uaij,ivpg,ja,uaij,vjag,ip->uvpg', V[o,v,o,o], T2, T1, Vint[o,v,o,o], T2, T1)
    #T2new -= np.einsum('uaij,ivpg,ja,uaij,viap,jg->uvpg', V[o,v,o,o], T2, T1, Vint[o,v,o,o], T2, T1)
    #T2new -= np.einsum('uaij,ivpg,ja,uaij,ijpg,va->uvpg', V[o,v,o,o], T2, T1, Vint[o,v,o,o], tau, T1)

    #T2new -= np.einsum('uaij,jvag,ip,uaij,vjag,ip->uvpg', V[o,v,o,o], T2, T1, Vint[o,v,o,o], T2, T1)
    #T2new -= np.einsum('uaij,jvag,ip,uaij,viap,jg->uvpg', V[o,v,o,o], T2, T1, Vint[o,v,o,o], T2, T1)
    #T2new -= np.einsum('uaij,jvag,ip,uaij,ijpg,va->uvpg', V[o,v,o,o], T2, T1, Vint[o,v,o,o], tau, T1)

    #T2new += np.einsum('abpi,uvag,ib->uvpg', V[v,v,v,o], T2, T1)
    #T2new += np.einsum('abpi,ivbg,ua->uvpg', V[v,v,v,o], T2, T1)

    #T2new -= np.einsum('abpi,ivgb,ua->uvpg', Vint[v,v,v,o], T2,T1)
    #T2new -= np.einsum('abpi,iuga,vb->uvpg', Vint[v,v,v,o], T2,T1)
    #T2new -= np.einsum('abpi,uvab,ig->uvpg', Vint[v,v,v,o], tau,T1)

    # END DEBUG

    T2new = T2new + np.einsum('uvpg->vugp',T2new)
    
    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D)
    
    res = np.sum(np.abs(T2new - T2))
    
    return T2new, res

def T1_iter(T1, T2):
    # Auxiliar (Intermediate) Arrays 
    
    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    V = 2*Vint - np.einsum('abij->baij', Vint)
    
    T1new  = -np.einsum('auip,ia->up', V[v,o,o,v], T1)
    
    Y = 2*T2 - np.einsum('ujpb->ujbp', tau)

    #X = np.einsum('ijpb,ua->ijupba',T2,T1) + np.einsum('ujab,ip->ijupba',T2,T1) - np.einsum('ujpb,ia->ijupba', Y, T1)

    #T1new += np.einsum('abij,ijupba->up', V[v,v,o,o], X)
    
    # No intermediate
    # DEBUG ###
    
    T1new += np.einsum('abij,ijpb,ua->up', V[v,v,o,o], T2, T1)
    T1new += np.einsum('abij,ujab,ip->up', V[v,v,o,o], T2, T1)
    T1new -= np.einsum('abij,ujpb,ia->up', V[v,v,o,o], Y, T1)
    # END DEBUG#

    T1new += np.einsum('auij,ijap->up', V[v,o,o,o], tau)
    
    T1new -= np.einsum('abip,iuab->up', V[v,v,o,v], tau)
    
    T1new = np.einsum('up,up->up', T1new, d)
    
    res = np.sum(np.abs(T1new - T1))
    
    return T1new, res

def Zap_T1_iter(T1, T2):
    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau)
    C1  = np.einsum('uaip,ia->uip', Vint[o,v,o,v], T1) #not sure here
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau)
    D1  = np.einsum('uapi,va->uvpi', Vint[o,v,v,o], T1)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2))
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2)
    Es1 = np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1)
    E1  = np.einsum('uaij,va->uvij', Vint[o,v,o,o], T1)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2))
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2)
    F11 = np.einsum('bapi,va->bvpi', Vint[v,v,v,o], T1)
    F12 = np.einsum('baip,va->bvip', Vint[v,v,o,v], T1)
    Fs1 = np.einsum('acpi,ib->acpb', Vint[v,v,v,o], T1)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2)) #careful
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau)

    X = E1 + D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2))
    
    X = Fs1 - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3))

    T1new = np.einsum('ui,ip->up', giu, T1)
    T1new -= np.einsum('ap,ua->up', gap, T1)
    T1new -= np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), T1, T1)
    T1new -= np.einsum('auip,ia->up', 2*(D2a - D2b) + D2c, T1)
    T1new -= np.einsum('aup->up', F2a)
    T1new += np.einsum('uiip->up', 1.0/2.0*(E2a - E2b) + E2c)
    T1new += np.einsum('uip->up', C1)
    T1new -= 2*np.einsum('uipi->up', D1)

    T1new = np.einsum('up,up->up', T1new, d)
    
    res = np.sum(np.abs(T1new - T1))
    
    return T1new, res

def Zap_T2_iter(T1, T2):
    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau)
    C1  = np.einsum('uaip,ia->uip', Vint[o,v,o,v], T1) #not sure here
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau)
    D1  = np.einsum('uapi,va->uvpi', Vint[o,v,v,o], T1)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2))
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2)
    Es1 = np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1)
    E1  = np.einsum('uaij,va->uvij', Vint[o,v,o,o], T1)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2))
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2)
    F11 = np.einsum('bapi,va->bvpi', Vint[v,v,v,o], T1)
    F12 = np.einsum('baip,va->bvip', Vint[v,v,o,v], T1)
    Fs1 = np.einsum('acpi,ib->acpb', Vint[v,v,v,o], T1)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2)) #careful
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau)

    X = E1 + D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2))
    
    X = Fs1 - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3))

    J = np.einsum('ag,uvpa->uvpg', gap, T2) - np.einsum('vi,uipg->uvpg', giu, T2)

    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1)

    S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  #notsure

    S += np.einsum('avig,uipa->uvpg', (D2a-D2b), T2 - Te.transpose(0,1,3,2))
    S += 0.5*np.einsum('avig,uipa->uvpg', D2c, T2)
    S += np.einsum('auig,viap->uvpg', D2c, Te)
    S += np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau)
    S -= np.einsum('uvpi,ig->uvpg', D1 + F2l, T1)
    S -= np.einsum('uvig,ip->uvpg',E2a - E2b - E2c.transpose(1,0,2,3), T1)
    S -= np.einsum('avgi,uipa->uvpg', F11, T2)
    S -= np.einsum('avpi,uiag->uvpg', F11, T2)
    S += np.einsum('avig,uipa->uvpg', F12, 2*T2 - T2.transpose(0,1,3,2))

    T2new = Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)

    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D)

    res2 = np.sum(np.abs(T2new - T2))
    
    return T2new, res2

def CCSD_Iter(T1, T2):

    # Intermediate arrays

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1)

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau)
    C1  = np.einsum('uaip,ia->uip', Vint[o,v,o,v], T1) #not sure here
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau)
    D1  = np.einsum('uapi,va->uvpi', Vint[o,v,v,o], T1)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2))
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2)
    Es1 = np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1)
    E1  = np.einsum('uaij,va->uvij', Vint[o,v,o,o], T1)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2))
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2)
    F11 = np.einsum('bapi,va->bvpi', Vint[v,v,v,o], T1)
    F12 = np.einsum('baip,va->bvip', Vint[v,v,o,v], T1)
    Fs1 = np.einsum('acpi,ib->acpb', Vint[v,v,v,o], T1)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2)) #careful
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau)

    X = E1 + D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2))
    
    X = Fs1 - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3))

    # T2 Amplitudes update

    J = np.einsum('ag,uvpa->uvpg', gap, T2) - np.einsum('vi,uipg->uvpg', giu, T2)

    S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  #notsure
    S += np.einsum('avig,uipa->uvpg', (D2a-D2b), T2 - Te.transpose(0,1,3,2))
    S += 0.5*np.einsum('avig,uipa->uvpg', D2c, T2)
    S += np.einsum('auig,viap->uvpg', D2c, Te)
    S += np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau)
    S -= np.einsum('uvpi,ig->uvpg', D1 + F2l, T1)
    S -= np.einsum('uvig,ip->uvpg',E2a - E2b - E2c.transpose(1,0,2,3), T1)
    S -= np.einsum('avgi,uipa->uvpg', F11, T2)
    S -= np.einsum('avpi,uiag->uvpg', F11, T2)
    S += np.einsum('avig,uipa->uvpg', F12, 2*T2 - T2.transpose(0,1,3,2))

    T2new = Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)

    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D)

    res2 = np.sum(np.abs(T2new - T2))

    # T1 Amplitudes update
    
    T1new = np.einsum('ui,ip->up', giu, T1)
    T1new -= np.einsum('ap,ua->up', gap, T1)
    T1new -= np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), T1, T1)
    T1new -= np.einsum('auip,ia->up', 2*(D2a - D2b) + D2c, T1)
    T1new -= np.einsum('aup->up', F2a)
    T1new += np.einsum('uiip->up', 1.0/2.0*(E2a - E2b) + E2c)
    T1new += np.einsum('uip->up', C1)
    T1new -= 2*np.einsum('uipi->up', D1)

    T1new = np.einsum('up,up->up', T1new, d)
    
    res1 = np.sum(np.abs(T1new - T1))

    return T1new, T2new, res1, res2

# Input Geometry    

#H2 = psi4.geometry("""
#    0 1
#    H 
#    H 1 0.76
#    symmetry c1
#""")

#water = psi4.geometry("""
#    0 1
#    O
#    H 1 0.96
#    H 1 0.96 2 104.5
#    symmetry c1
#""")
#
form = psi4.geometry("""
0 1
O
C 1 1.22
H 2 1.08 1 120.0
H 2 1.08 1 120.0 3 -180.0
symmetry c1
""")

# Basis set

basis = 'cc-pvdz'

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': basis,
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

nelec = wfn.nalpha() + wfn.nbeta()
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
nvir = nmo - ndocc
eps = np.asarray(wfn.epsilon_a())
nbf = C.shape[0]

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Electrons:            {}".format(nelec))
print("Number of Molecular Orbitals:   {}".format(nmo))
print("Number of Doubly ocuppied MOs:  {}".format(ndocc))

# Get Integrals

print("Converting atomic integrals to MO integrals...")
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
Vint = np.asarray(mints.mo_eri(C, C, C, C))
# Convert to physicist notation
Vint = Vint.swapaxes(1,2)
print("Completed in {} seconds!".format(time.time()-t))

# Slices

o = slice(0, ndocc)
v = slice(ndocc, nbf)

# START CCSD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrix...')
t = time.time()
D  = np.zeros([ndocc, ndocc, nvir, nvir])
d  = np.zeros([ndocc, nvir])
for i,ei in enumerate(eps[o]):
    for j,ej in enumerate(eps[o]):
        for a,ea in enumerate(eps[v]):
            d[i,a] = 1/(ea - ei)
            for b,eb in enumerate(eps[v]):
                D[i,j,a,b] = 1/(ei + ej - ea - eb)

print('Done. Time required: {:.5f} seconds'.format(time.time() - t))

print('\nComputing MP2 guess')

t = time.time()

T1 = np.zeros([ndocc, nvir])
T2 = np.einsum('abij,ijab->ijab', Vint[v,v,o,o], D)

E = cc_energy(T1, T2)

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
    T1, T2, r1, r2 = CCSD_Iter(T1, T2)
    E = cc_energy(T1, T2)
    dE = E - Eold
    print('-'*50)
    print("Iteration {}".format(ite))
    print("CC Correlation energy: {}".format(E))
    print("Energy change:         {}".format(dE))
    print("T1 Residue:            {}".format(r1))
    print("T2 Residue:            {}".format(r2))
    print("Max T1 Amplitude:      {}".format(np.max(T1)))
    print("Max T2 Amplitude:      {}".format(np.max(T2)))
    print("Time required:         {}".format(time.time() - t))
    print('-'*50)

print("\nCC Equations Converged!!!")
print("Final CCSD Energy:     {:<5.10f}".format(E + scf_e))
print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
print("Total Computation time:        {}".format(time.time() - tinit))


