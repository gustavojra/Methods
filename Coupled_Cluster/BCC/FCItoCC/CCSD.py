import psi4
import os
import sys
import numpy as np
import time
import copy

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *

np.set_printoptions(suppress=True)

### FUNCTIONS ###

def cc_energy(T1, T2, Vint, fo, o, v):

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    X = 2*tau - np.einsum('ijab->jiab',tau)
    E = 2*np.einsum('ia,ia->', fo[o,v], T1) + np.einsum('abij,ijab->', Vint[v,v,o,o], X)
    return E

def update_amp(T1, T2, Vint, d, D, fo, o, v, EINSUMOPT='optimal'):

    # Intermediate arrays

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1,optimize=EINSUMOPT)
    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1,optimize=EINSUMOPT)

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau,optimize=EINSUMOPT)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau,optimize=EINSUMOPT)
    C1  = np.einsum('uaip,ia->uip', Vint[o,v,o,v], T1,optimize=EINSUMOPT) 
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2,optimize=EINSUMOPT)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau,optimize=EINSUMOPT)
    D1  = np.einsum('uapi,va->uvpi', Vint[o,v,v,o], T1,optimize=EINSUMOPT)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    Es1 = np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1,optimize=EINSUMOPT)
    E1  = np.einsum('uaij,va->uvij', Vint[o,v,o,o], T1,optimize=EINSUMOPT)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    F11 = np.einsum('bapi,va->bvpi', Vint[v,v,v,o], T1,optimize=EINSUMOPT)
    F12 = np.einsum('baip,va->bvip', Vint[v,v,o,v], T1,optimize=EINSUMOPT)
    Fs1 = np.einsum('acpi,ib->acpb', Vint[v,v,v,o], T1,optimize=EINSUMOPT)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT) 
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau,optimize=EINSUMOPT)

    X = E1 + D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2),optimize=EINSUMOPT)
    
    X = Fs1 - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3),optimize=EINSUMOPT)

    # T2 Amplitudes update

    J = np.einsum('ap,uvag->uvpg',fo[v,v],T2) - np.einsum('ui,ivpg->uvpg',fo[o,o],T2) 
    J += np.einsum('ai,uvag,ip->uvpg',fo[v,o],T2,T1) 
    J += np.einsum('ai,uipg,va->uvpg',fo[v,o],T2,T1) 
    J += np.einsum('ag,uvpa->uvpg', gap, T2,optimize=EINSUMOPT) - np.einsum('vi,uipg->uvpg', giu, T2,optimize=EINSUMOPT)

    S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  
    S += np.einsum('avig,uipa->uvpg', (D2a-D2b), T2 - Te.transpose(0,1,3,2),optimize=EINSUMOPT)
    S += 0.5*np.einsum('avig,uipa->uvpg', D2c, T2,optimize=EINSUMOPT)
    S += np.einsum('auig,viap->uvpg', D2c, Te,optimize=EINSUMOPT)
    S += np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau,optimize=EINSUMOPT)
    S -= np.einsum('uvpi,ig->uvpg', D1 + F2l, T1,optimize=EINSUMOPT)
    S -= np.einsum('uvig,ip->uvpg',E2a - E2b - E2c.transpose(1,0,2,3), T1,optimize=EINSUMOPT)
    S -= np.einsum('avgi,uipa->uvpg', F11, T2,optimize=EINSUMOPT)
    S -= np.einsum('avpi,uiag->uvpg', F11, T2,optimize=EINSUMOPT)
    S += np.einsum('avig,uipa->uvpg', F12, 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)

    T2new = Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)

    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D,optimize=EINSUMOPT)

    res2 = np.sum(np.abs(T2new - T2))

    # T1 Amplitudes update
    T1new = -fo[o,v] + np.einsum('ui,ip->up', fo[o,o], T1) 
    T1new += -np.einsum('ap,ua->up', fo[v,v], T1) 
    T1new += 2*np.einsum('ai,uipa->up',fo[v,o],T2) - np.einsum('ai,uiap->up', fo[v,o], tau)
    
    T1new = np.einsum('ui,ip->up', giu, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('ap,ua->up', gap, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), T1, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('auip,ia->up', 2*(D2a - D2b) + D2c, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('aup->up', F2a,optimize=EINSUMOPT)
    T1new += np.einsum('uiip->up', 1.0/2.0*(E2a - E2b) + E2c,optimize=EINSUMOPT)
    T1new += np.einsum('uip->up', C1,optimize=EINSUMOPT)
    T1new -= 2*np.einsum('uipi->up', D1,optimize=EINSUMOPT)

    T1new = np.einsum('up,up->up', T1new, d,optimize=EINSUMOPT)
    
    res1 = np.sum(np.abs(T1new - T1))

    return T1new, T2new, res1, res2


def CCSD(Vint, fd, fo, ndocc, nvir, scf_e):
    
    print("SCF energy: {}".format(scf_e))
    nbf = ndocc + nvir

    # Slices
    
    o = slice(0, ndocc)
    v = slice(ndocc, nbf)
    
    # START CCSD CODE
    
    # Build the Auxiliar Matrix D
    
    print('\n----------------- RUNNING CCD ------------------')
    
    print('\nBuilding Auxiliar D matrix...')
    D  = np.zeros([ndocc, ndocc, nvir, nvir])
    d  = np.zeros([ndocc, nvir])
    for i,ei in enumerate(fd[o]):
        for j,ej in enumerate(fd[o]):
            for a,ea in enumerate(fd[v]):
                d[i,a] = 1/(ea - ei)
                for b,eb in enumerate(fd[v]):
                    D[i,j,a,b] = 1/(ei + ej - ea - eb)
    
    print('\nComputing MP2 guess')
    
    T1 = np.zeros([ndocc, nvir])
    T2 = np.einsum('abij,ijab->ijab', Vint[v,v,o,o], D)
    
    E = cc_energy(T1, T2, Vint, fo, o, v)
    
    print('MP2 Energy: {:<5.10f}'.format(E+scf_e))
    
    r1 = 0
    r2 = 1
    CC_CONV = 8
    CC_MAXITER = 30
        
    LIM = 10**(-CC_CONV)
    
    ite = 0
    
    while r2 > LIM or r1 > LIM:
        ite += 1
        if ite > CC_MAXITER:
            raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
        Eold = E
        t = time.time()
        T1, T2, r1, r2 = update_amp(T1, T2, Vint, d, D, fo, o, v)
        E = cc_energy(T1, T2, Vint, fo, o, v)
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
