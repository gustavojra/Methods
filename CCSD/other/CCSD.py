import psi4
import numpy as np
import scipy.linalg as la
import os
import sys
import timeit
import copy

file_dir = os.path.dirname('../Aux/')
sys.path.append(file_dir)

file_dir = os.path.dirname('../HF/RHF/')
sys.path.append(file_dir)

from tools import *
from rhf import RHF

class CC:

# Compute Coupled Cluster Energies

    def __init__(self, HF):
        self.E0 = HF.E
        self.orbitals = HF.orbitals
        self.ndocc = HF.ndocc
        self.nelec = HF.nelec
        self.nbf = HF.nbf
        self.virtual = self.nbf - self.ndocc
        self.V_nuc = HF.V_nuc
        self.h = HF.T + HF.V
        self.F = HF.Fmol
        self.eo = HF.Eorb
        self.holes = range(0, self.ndocc)
        self.particles = range(self.ndocc, self.nbf)
        print("Number of electrons: {}".format(self.nelec))
        print("Number of basis functions: {}".format(self.nbf))
        print("Number of doubly occupied orbitals: {}".format(self.ndocc))
        print("Number of virtual spatial orbitals: {}".format(self.virtual))

# Convert atomic integrals to MO integrals

        psi4_orb = psi4.core.Matrix.from_array(self.orbitals)
        print("Converting atomic integrals to MO integrals...")
        self.MIone = np.einsum('up,vq,uv->pq', self.orbitals, self.orbitals, self.h)
        self.MItwo = np.asarray(HF.mints.mo_eri(psi4_orb, psi4_orb, psi4_orb, psi4_orb))
        # Convert to physicist notation
        self.MItwo = self.MItwo.swapaxes(1,2)
        self.antis = self.MItwo - self.MItwo.swapaxes(2,3) 
        print("Completed!")

# Function to compute energy given a set of amplitudes

    def CC_Energy(self, T1, T2):

        tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
        X = 2*tau - np.einsum('ijab->jiab',tau)
        E = np.einsum('ijab,ijab->',self.MItwo,X)
        return E
        
    def Iter_T1(self, T1, T2):

        # Auxiliar (Intermediate) Arrays 

        tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
        V = 2*self.MItwo - np.einsum('ijab->ijba',self.MItwo)

        T1new  = -np.einsum('ipau,ia->up', V, T1)
        
        X = 2*T2 - tau.swapaxes(2,3) 

        X = np.einsum('ijpb,ua->ijupba',T2,T1) + np.einsum('ujab,ip->ijupba',T2,T1) - np.einsum('ujpb,ia->ijupba', X, T1)
        
        T1new += np.einsum('ijab,ijupba->up', V, X)

        T1new += np.einsum('ijau,ijap->up', V, tau)

        T1new -= np.einsum('ipab,iuab->up', V, tau)

        T1new = np.einsum('up,up->up', T1new, self.d)

        res = np.sum(np.abs(T1new - T1))

        return T1new, res

    def Iter_T2(self, T1, T2):

        # Auxiliar (Intermediate) Arrays 

        tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
        V = 2*self.MItwo - np.einsum('ijab->ijba',self.MItwo)
        Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1)

        T2new = copy.deepcopy(self.MItwo)
        
        T2new += np.einsum('ijuv,ijpg->uvpg', self.MItwo, tau)

        T2new += np.einsum('pgab,uvab->uvpg', self.MItwo, tau)

        T2new += np.einsum('ijab,ijpg,uvab->uvpg', self.MItwo, tau, tau)

        T2new = 0.5*T2new

        T2new += np.einsum('pgua,va->uvpg', self.MItwo, T1)

        T2new -= np.einsum('piuv,ig->uvpg', self.MItwo, T1)

        T2new += np.einsum('ipau,viga->uvpg', V, T2)

        T2new -= np.einsum('igua,ivpa->uvpg', self.MItwo, tau)

        T2new -= np.einsum('ipau,viag->uvpg', self.MItwo, tau)

        X = T2 - np.einsum('uipa->uiap',tau)

        T2new += np.einsum('ijab,vjgb,uipa->uvpg', V, T2, X)

        T2new -= np.einsum('ijab,ijgb,uvpa->uvpg', V, tau, T2)

        T2new -= np.einsum('ijab,vjab,uipg->uvpg', V, tau, T2)

        T2new += np.einsum('ijab,vjbg,uiap->uvpg', self.MItwo, T2, Te)

        T2new += np.einsum('ijab,ujag,vibp->uvpg', self.MItwo, T2, Te)

        X = np.einsum('ivpg,ja->ivjpga', T2, T1) + np.einsum('jvag,ip->ivjpga', T2, T1)

        Y = np.einsum('vjag,ip->vjiagp', T2, T1) + np.einsum('viap,jg->vjiagp', T2, T1) + np.einsum('ijpg,va->vjiagp', tau, T1)

        T2new -= np.einsum('ijua,ivjpga,ijua,vjiagp->uvpg', V, X, self.MItwo, Y)

        X = np.einsum('uvag,ib->uviagb', T2, T1) + np.einsum('ivbg,ua->uviagb', T2, T1)

        T2new += np.einsum('piab,uviagb->uvpg', V, X)

        Y = np.einsum('ivgb,ua->ivugba', T2, T1) + np.einsum('iuga,vb->ivugba', T2, T1) + np.einsum('uvab,ig->ivugba', tau, T1)

        T2new -= np.einsum('piab,ivugba->uvpg', self.MItwo, Y)

        # Finish it up: Apply the permutation operator and divide by orbital energies [D(ijab)]

        T2new = T2new + np.einsum('uvpg->vugp',T2new)

        T2new = np.einsum('uvpg,uvpg->uvpg', T2new, self.D)

        res = np.sum(np.abs(T2new - T2))

        return T2new, res
        
        
    def CCSD(self, CC_CONV=6, CC_MAXITER=50):

        # Build auxiliar D and d  matrices for T2 and T1 amplitudes, respectivamente.

        self.D = np.zeros([self.nbf, self.nbf, self.nbf, self.nbf])
        self.d = np.zeros([self.nbf, self.nbf])
        for i in self.holes:
            for a in self.particles:
                self.d[i,a] = 1/(self.eo[i] - self.eo[a])
                for j in self.holes:
                    for b in self.particles:
                        self.D[i,j,a,b] = 1/(self.eo[i] + self.eo[j] - self.eo[a] - self.eo[b])

        # Compute initial guess for T1 and T2 amplitudes

        T1 = np.zeros([self.nbf, self.nbf])
        T2 = np.einsum('ijab,ijab->ijab', self.MItwo, self.D)
        
        # Report the MP2 Energy from the initial guess
        Emp2 = self.CC_Energy(T1, T2) + self.E0
        psi4.core.print_out('CC MP2 Energy:    {:<5.10f}\n'.format(Emp2))
        
        ite = 1
        T1,r1 = self.Iter_T1(T1, T2)
        T2,r2 = self.Iter_T2(T1, T2)
        E = self.CC_Energy(T1, T2)
        print('-'*50)
        print("Iteration {}".format(ite))
        print("CC Correlation energy: {}".format(E))
        print("T1 Residue: {}".format(r1))
        print("T2 Residue: {}".format(r2))
        print('-'*50)

        LIM = 10**(-CC_CONV)

        while r1 > LIM or r2 > LIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            T1,r1 = self.Iter_T1(T1, T2)
            T2,r2 = self.Iter_T2(T1, T2)
            E = self.CC_Energy(T1, T2)
            print('-'*50)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {}".format(E))
            print("T1 Residue: {}".format(r1))
            print("T2 Residue: {}".format(r2))
            print('-'*50)
        
        print("\nCC Equations Converged!!!")
        print("Final CC Energy: {}".format(E + self.E0))


    def CCD(self, CC_CONV=6, CC_MAXITER=50):

        # Build auxiliar D and d  matrices for T2 and T1 amplitudes, respectivamente.

        self.D = np.zeros([self.nbf, self.nbf, self.nbf, self.nbf])
        self.d = np.zeros([self.nbf, self.nbf])
        for i in self.holes:
            for a in self.particles:
                self.d[i,a] = 1/(self.eo[i] - self.eo[a])
                for j in self.holes:
                    for b in self.particles:
                        self.D[i,j,a,b] = 1/(self.eo[i] + self.eo[j] - self.eo[a] - self.eo[b])

        # Compute initial guess for T2 amplitudes

        T1 = np.zeros([self.nbf, self.nbf])
        T2 = np.einsum('ijab,ijab->ijab', self.MItwo, self.D)
        
        # Report the MP2 Energy from the initial guess
        Emp2 = self.CC_Energy(T1, T2) + self.E0
        psi4.core.print_out('CC MP2 Energy:    {:<5.10f}\n'.format(Emp2))
        
        ite = 1
        T2,r2 = self.Iter_T2(T1, T2)
        E = self.CC_Energy(T1, T2)
        print('-'*50)
        print("Iteration {}".format(ite))
        print("CC Correlation energy: {}".format(E))
        print("T2 Residue: {}".format(r2))
        print('-'*50)

        LIM = 10**(-CC_CONV)

        while r2 > LIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            T2,r2 = self.Iter_T2(T1, T2)
            E = self.CC_Energy(T1, T2)
            print('-'*50)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {}".format(E))
            print("T2 Residue: {}".format(r2))
            print('-'*50)
        
        print("\nCC Equations Converged!!!")
        print("Final CCD Energy: {}".format(E + self.E0))
