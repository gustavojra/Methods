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

def cc_energy(T2, Vint, fo, o, v):

    X = 2*T2 - np.einsum('ijab->jiab',T2)
    E = np.einsum('abij,ijab->', Vint[v,v,o,o], X)
    return E

def update_amp(T2, Vint, D, fo, o, v, EINSUMOPT='optimal'):

    # Intermediate arrays

    tau = T2 
    Te = 0.5*T2 

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau,optimize=EINSUMOPT)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau,optimize=EINSUMOPT)
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2,optimize=EINSUMOPT)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau,optimize=EINSUMOPT)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT) 
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau,optimize=EINSUMOPT)

    X = D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2),optimize=EINSUMOPT)
    
    X = - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3),optimize=EINSUMOPT)

    # T2 Amplitudes update

    J = np.einsum('ap,uvag->uvpg',fo[v,v],T2) - np.einsum('ui,ivpg->uvpg',fo[o,o],T2) 
    J += np.einsum('ag,uvpa->uvpg', gap, T2,optimize=EINSUMOPT) - np.einsum('vi,uipg->uvpg', giu, T2,optimize=EINSUMOPT)

    S = 0.5*A2l + 0.5*B2l - (C2 + C2l - D2a).transpose(2,1,0,3)  
    S += np.einsum('avig,uipa->uvpg', (D2a-D2b), T2 - Te.transpose(0,1,3,2),optimize=EINSUMOPT)
    S += 0.5*np.einsum('avig,uipa->uvpg', D2c, T2,optimize=EINSUMOPT)
    S += np.einsum('auig,viap->uvpg', D2c, Te,optimize=EINSUMOPT)
    S += np.einsum('uvij,ijpg->uvpg', 0.5*D2l, tau,optimize=EINSUMOPT)

    T2new = Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)

    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D,optimize=EINSUMOPT)

    res2 = np.sum(np.abs(T2new - T2))

    return T2new, res2


def CCD(Vint, fd, fo, ndocc, nvir, scf_e):
    
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
    for i,ei in enumerate(fd[o]):
        for j,ej in enumerate(fd[o]):
            for a,ea in enumerate(fd[v]):
                for b,eb in enumerate(fd[v]):
                    D[i,j,a,b] = 1/(ei + ej - ea - eb)
    
    print('\nComputing MP2 guess')
    
    T2 = np.einsum('abij,ijab->ijab', Vint[v,v,o,o], D)
    
    E = cc_energy(T2, Vint, fo, o, v)
    
    print('MP2 Energy: {:<5.10f}'.format(E+scf_e))
    
    r2 = 1
    CC_CONV = 8
    CC_MAXITER = 30
        
    LIM = 10**(-CC_CONV)
    
    ite = 0
    
    while r2 > LIM:
        ite += 1
        if ite > CC_MAXITER:
            raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
        Eold = E
        t = time.time()
        T2, r2 = update_amp(T2, Vint, D, fo, o, v)
        E = cc_energy(T2, Vint, fo, o, v)
        dE = E - Eold
        print('-'*50)
        print("Iteration {}".format(ite))
        print("CC Correlation energy: {}".format(E))
        print("Energy change:         {}".format(dE))
        print("T2 Residue:            {}".format(r2))
        print("Max T2 Amplitude:      {}".format(np.max(T2)))
        print("Time required:         {}".format(time.time() - t))
        print('-'*50)
    
    print("\nCC Equations Converged!!!")
    print("Final CCSD Energy:     {:<5.10f}".format(E + scf_e))
