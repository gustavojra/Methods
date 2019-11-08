import psi4
import numpy as np
import scipy.linalg as la
import os
import sys
import timeit

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

    def CC_Energy(self, T1SS, T1OS, T2SS, T2OS):

        #tau = T2OS + np.einsum('ia,jb->ijab', T1SS, T1SS)
        #X = 2*tau - tau.swapaxes(0,1)
        #return np.einsum('ijab,ijab->',self.MItwo,X)

        e1 = np.einsum('ijab, ia, jb->', self.antis, T1SS, T1SS) + 2*np.einsum('ijab, ia, jb->', self.MItwo, T1OS, T1OS)
        e2 = 0.5*np.einsum('ijab,ijab->', self.antis, T2SS) + np.einsum('ijab,ijab->', self.MItwo, T2OS)
        return e1 + e2
        
    def Iter_T1(self, T1SS, T1OS, T2SS, T2OS):
        
        T1SS_out = np.zeros ([self.nbf, self.nbf])

        # Update T1 Same Spin, orbitals a and i assumed to be alpha.
        
        # Terms in the same order as shown in pag 75 Crawford and Schaefer, Rev Comp Chem Vol 14 Chap 2

        # First term F(a,i) zero due to cannonical HF orbitals

        # Second and third terms are simply the matrix element D(i,a) times T(a,i), those are transfered to the left side of the equation

        # 4th term for the case all alphas   !!! COULD MERGE THOSE TWO EINSUMS
        hold = np.einsum('kaci,kc->ia', self.antis, T1SS)
        # 4th term for the case k, c betas
        hold += np.einsum('kaci,kc->ia', self.MItwo, T1SS)

        # 5th term is zero due cannocal HF orbitals

        # 6th term for all alpha  !! Maybe merging would help?
        hold += 0.5*np.einsum('kacd,kicd->ia', self.antis, T2SS)
        # 6th term for k and c betas or k and d betas (equivalent cases, thus 2x factor)
        hold += np.einsum('kacd,kicd->ia', self.MItwo, T2OS)

        # 7th term all alphas  !! Maybe merging would help?
        hold -= 0.5*np.einsum('klci,klca->ia', self.antis, T2SS)
        # 7th term k and c betas or l and c betas (eq. cases, thus 2x factor)
        hold -= np.einsum('klci,klca->ia', self.MItwo, T2OS)

        # 8th term zero due cannonical HF orbitals

        # 9th term all alphas
        hold -= np.einsum('klci,kc,la->ia', self.antis, T1SS, T1SS)
        # 9th term k and c betas or l and c betas (eq. cases, thus 2x factor)
        # PRODUCT OF T1SS ALPHA AND T1SS BETA = 0????
        #hold -= 2*np.einsum('klci,kc,la->ia', self.MItwo, T1SS, T1SS)

        # 10th term all alphas
        hold -= np.einsum('kacd,kc,id->ia', self.antis, T1SS, T1SS)
        # 10th term k and c betas or k and d betas (eq. cases, thus 2x factor)
        # PRODUCT OF T1SS ALPHA AND T1SS BETA = 0????
        #hold -= 2*np.einsum('kacd,kc,id->ia', self.MItwo, T1SS, T1SS)

        # 11th term all alphas
        hold -= np.einsum('klcd,kc,id,la->ia', self.antis, T1SS, T1SS, T1SS)
        # PRODUCT OF T1SS ALPHA AND T1SS BETA = 0????
       # # 11th term all betas
       # hold -= np.einsum('klcd,kc,id,la->ia', self.antis, T1SS, T1OS, T1OS)
       # # 11th term k and c beta and k and d beta (eq. cases, thus 2x factor)
       # hold -= 2*np.einsum('klcd,kc,id,la->ia', self.MItwo, T1SS, T1SS, T1SS)
       # # 11th term l and d beta and l and c beta (eq. cases, thus 2x factor)
       # hold -= 2*np.einsum('klcd,kc,id,la->ia', self.MItwo, T1SS, T1OS, T1OS)

        T1SS_out[i,a] = d[i,a]*hold
                
        

    def SDT2(self, T1_init, T2_init):
        pass

    def CCSD(self, CC_CONV=6, CC_MAXITER=50):

        # Build auxiliar D and d  matrices for T2 and T1 amplitudes, respectivamente.

        D = np.zeros([self.nbf, self.nbf, self.nbf, self.nbf])
        d = np.zeros([self.nbf, self.nbf])
        for i in self.holes:
            for a in self.particles:
                d[i,a] = 1/(self.eo[i] - self.eo[a])
                for j in self.holes:
                    for b in self.particles:
                        D[i,j,a,b] = 1/(self.eo[i] + self.eo[j] - self.eo[a] - self.eo[b])

        # Compute initial guess for T1 and T2 amplitudes

        T1SS = np.zeros([self.nbf, self.nbf])
        T1OS = np.zeros([self.nbf, self.nbf])

        T2SS = np.einsum('ijab,ijab->ijab', self.antis, D)
        T2OS = np.einsum('ijab,ijab->ijab', self.MItwo, D)
        
        # Report the MP2 Energy from the initial guess
        Emp2 = self.CC_Energy(T1SS, T1OS, T2SS, T2OS) + self.E0
        psi4.core.print_out('CC MP2 Energy:    {:<5.10f}\n'.format(Emp2))
