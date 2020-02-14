import psi4
import os
import sys
import numpy as np
import time
import copy

sys.path.append('../../Aux')
from tools import *

np.set_printoptions(suppress=True, linewidth=120)

class RCCSD:

    def update_energy(self):
        
        X = 2*self.T2 + 2*np.einsum('IA,JB->IJAB', self.T1, self.T1,optimize='optimal')
        X += - self.T2.transpose(1,0,2,3) - np.einsum('JA,IB->IJAB', self.T1, self.T1,optimize='optimal')
        self.Ecc = np.einsum('IjAb,IjAb->', X, self.Voovv,optimize='optimal')

        #X = 2*self.T2 - self.T2.transpose(1,0,2,3)
        #self.Ecc = np.einsum('IjAb,IjAb->', X, self.Voovv,optimize='optimal')

    def update_tau_and_te(self):

        self.tau = self.T2 + 0.5*np.einsum('IA,JB->IJAB', self.T1, self.T1,optimize='optimal')
        self.Te = self.T2 + np.einsum('IA,JB->IJAB', self.T1, self.T1,optimize='optimal')
    
    def update_Fint(self):

        # Update F(AE)
        self.Fae = np.zeros((self.nvir, self.nvir))
        self.Fae += self.fock_VV - 0.5*np.einsum('ME,MA->AE', self.fock_OV, self.T1,optimize='optimal')
        self.Fae += np.einsum('MF,MAFE->AE', self.T1, self.Aovvv,optimize='optimal')
        self.Fae += -np.einsum('MnAf,MnEf->AE', self.tau, self.Aoovv,optimize='optimal')

        # Update F(MI)
        self.Fmi = np.zeros((self.ndocc, self.ndocc))
        self.Fmi += self.fock_OO + 0.5*np.einsum('ME,IE->MI', self.fock_OV, self.T1,optimize='optimal')
        self.Fmi += np.einsum('NE,MNIE->MI', self.T1, self.Aooov,optimize='optimal')
        self.Fmi += np.einsum('INEF,MNEF->MI', self.tau, self.Aoovv,optimize='optimal')

        # Update F(ME)
        self.Fme = np.zeros((self.ndocc, self.nvir))
        self.Fme += self.fock_OV + np.einsum('NF, MNEF-> ME', self.T1, self.Aoovv,optimize='optimal')

    def update_Winf(self):

        # Update W(MnIj)
        self.Wmnij = np.zeros((self.ndocc, self.ndocc, self.ndocc, self.ndocc))
        self.Wmnij += self.Voooo
        self.Wmnij += np.einsum('je, MnIe-> MnIj', self.T1, self.Vooov,optimize='optimal')
        self.Wmnij += np.einsum('IE,nMjE -> MnIj', self.T1, self.Vooov,optimize='optimal')
        self.Wmnij += (1.0/2.0)*np.einsum('IjEf,MnEf->MnIj', self.Te, self.Voovv,optimize='optimal')

        # Update W(AbEf)
        self.Wabef = np.zeros((self.nvir, self.nvir, self.nvir, self.nvir))
        self.Wabef += self.Vvvvv
        self.Wabef += -np.einsum('mb, mAfE-> AbEf', self.T1, self.Vovvv,optimize='optimal')
        self.Wabef += -np.einsum('MA, MbEf -> AbEf', self.T1, self.Vovvv,optimize='optimal')
        self.Wabef += (1.0/2.0)*np.einsum('MnAb,MnEf->AbEf', self.Te, self.Voovv,optimize='optimal')

        # Update W(MbEj)
        self.W_MbEj = np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))
        self.W_MbEj += self.Voovv.transpose(0,3,2,1)
        self.W_MbEj += np.einsum('jf,MbEf->MbEj', self.T1, self.Vovvv,optimize='optimal')
        self.W_MbEj += -np.einsum('nb,nMjE->MbEj', self.T1, self.Vooov,optimize='optimal')
        X = self.T2 + 2*np.einsum('jf,nb->jnfb', self.T1, self.T1,optimize='optimal')
        self.W_MbEj += -0.5*np.einsum('jnfb,MnEf->MbEj', X, self.Voovv,optimize='optimal')
        self.W_MbEj += 0.5*np.einsum('NjFb,MNEF->MbEj', self.T2, self.Aoovv,optimize='optimal')

        # Update W(MbeJ)
        self.W_MbeJ = np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))
        self.W_MbeJ += -self.Vovov.transpose(0,1,3,2)
        self.W_MbeJ += -np.einsum('JF,MbFe->MbeJ', self.T1, self.Vovvv,optimize='optimal')
        self.W_MbeJ += np.einsum('nb,MnJe->MbeJ', self.T1, self.Vooov,optimize='optimal')
        X = 0.5*self.T2 + np.einsum('JF,nb->JnFb', self.T1, self.T1,optimize='optimal')
        self.W_MbeJ += np.einsum('JnFb,nMeF->MbeJ', X, self.Voovv,optimize='optimal')

    def update_amp(self):

        # Create a new set of amplitudes
        newT1 = np.zeros(self.T1.shape)
        newT2 = np.zeros(self.T2.shape)

        # Update T(IA)
        newT1 += self.fock_OV 
        newT1 += np.einsum('IE,AE->IA', self.T1, self.Fae,optimize='optimal')
        newT1 += -np.einsum('MA,MI->IA', self.T1, self.Fmi,optimize='optimal')
        X = 2*self.T2 - self.T2.transpose(1,0,2,3)
        newT1 += np.einsum('ImAe,me->IA', X, self.Fme,optimize='optimal')
        newT1 += np.einsum('ME,AMIE->IA', self.T1, self.Avoov,optimize='optimal')
        newT1 += -np.einsum('MnAe,MnIe->IA', self.T2, self.Aooov,optimize='optimal')
        newT1 += np.einsum('ImEf,mAfE->IA', self.T2, self.Aovvv,optimize='optimal')
        newT1 *= self.d

        # Update T(IjAb)

        newT2 += self.Voovv
        newT2 += np.einsum('mnab,mnij->ijab', self.Te, self.Wmnij,optimize='optimal')
        newT2 += np.einsum('ijef,abef->ijab', self.Te, self.Wabef,optimize='optimal')

        X = self.Fae - 0.5*np.einsum('mb,me->be', self.T1, self.Fme,optimize='optimal')
        P = np.einsum('IjAe,be->IjAb', self.T2, X,optimize='optimal')
        X = self.Fmi + 0.5*np.einsum('je,me->mj', self.T1, self.Fme,optimize='optimal')
        P += - np.einsum('ImAb,mj->IjAb', self.T2, X,optimize='optimal')
        
        X = self.T2 - self.T2.transpose(1,0,2,3)
        P += np.einsum('imae,mbej->ijab', X, self.W_MbEj,optimize='optimal')
        P += -np.einsum('ie,ma,mbej->ijab', self.T1, self.T1, self.Voovv.transpose(0,3,2,1),optimize='optimal')

        P += np.einsum('imae,mbej->ijab', self.T2, self.W_MbEj + self.W_MbeJ,optimize='optimal')

        P += np.einsum('mibe,maej->ijab', self.T2, self.W_MbeJ,optimize='optimal')
        P += -np.einsum('ie,mb,maje->ijab', self.T1, self.T1, self.Vovov,optimize='optimal')

        P += np.einsum('IE,jEbA->IjAb', self.T1, self.Vovvv,optimize='optimal')
        P += -np.einsum('MA,IjMb->IjAb', self.T1, self.Vooov,optimize='optimal')

        newT2 += P + P.transpose(1,0,3,2)

        newT2 *= self.D

        # Compute RMS

        newT1[self.activeO, self.activeV] = 0.0
        self.rms1 = np.sqrt(np.sum(np.square(newT1 - self.T1 )))/(self.ndocc*self.nvir)
        self.rms2 = np.sqrt(np.sum(np.square(newT2 - self.T2 )))/(self.ndocc*self.ndocc*self.nvir*self.nvir)

        # Save new amplitudes

        self.T1 = newT1
        self.T2 = newT2

    def __init__(self, orbitals, scfe, nmo, nalpha, nbeta, Vnuc, mints, nactive, nfrozen, T1=False, T2=False, T3=False, T4a=False, T4b=False, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

        # Save reference wavefunction properties
        self.Ehf = scfe
        self.nmo = nmo
        self.nelec = nalpha + nbeta
        if self.nelec % 2 != 0:
            NameError('Invalid number of electrons for RHF') 
        self.ndocc = int(self.nelec/2)
        self.nvir = self.nmo - self.ndocc
        self.C = psi4.core.Matrix.from_array(orbitals)
        self.Vnuc = Vnuc
        self.activeO = slice(nfrozen, self.ndocc)
        self.activeV = slice(0, self.nvir - self.nmo +  nactive + nfrozen)

        # Save Options
        self.CC_CONV = CC_CONV
        self.E_CONV = E_CONV
        self.CC_MAXITER = CC_MAXITER

        # Input amplitudes
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4abab = T4a
        self.T4abaa = T4b

        print("Number of electrons:              {}".format(self.nelec))
        print("Number of Doubly Occupied MOs:    {}".format(self.ndocc))
        print("Number of MOs:                    {}".format(self.nmo))
        print("Number of Active Occ:             {}".format(self.activeO))
        print("Number of Active Vir:             {}".format(self.activeV))

        print("\n Transforming integrals...")

        self.mints = mints
        # One electron integral
        h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        ## Alpha
        h = np.einsum('up,vq,uv->pq', self.C, self.C, h)
        Vchem = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
    
        # Slices
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nmo)

        # Form the full fock matrices
        f = h + 2*np.einsum('pqkk->pq', Vchem[:,:,o,o]) - np.einsum('pkqk->pq', Vchem[:,o,:,o])

        # Save diagonal terms
        self.fock_Od = copy.deepcopy(f.diagonal()[o])
        self.fock_Vd = copy.deepcopy(f.diagonal()[v])

        # Erase diagonal elements from original matrix
        np.fill_diagonal(f, 0.0)

        # Save useful slices
        self.fock_OO = f[o,o]
        self.fock_VV = f[v,v]
        self.fock_OV = f[o,v]

        # Save slices of two-electron repulsion integral
        Vphys = Vchem.swapaxes(1,2)
        Vsa = 2*Vphys - Vphys.swapaxes(2,3)

        self.Aovvv = Vsa[o,v,v,v]
        self.Aooov = Vsa[o,o,o,v]
        self.Aoovv = Vsa[o,o,v,v]
        self.Avoov = Vsa[v,o,o,v]

        self.Voooo = Vphys[o,o,o,o]
        self.Vooov = Vphys[o,o,o,v]
        self.Voovv = Vphys[o,o,v,v]
        self.Vovov = Vphys[o,v,o,v]
        self.Vovvv = Vphys[o,v,v,v]
        self.Vvvvv = Vphys[v,v,v,v]

        self.compute()

    def compute(self):

        # Auxiliar D matrices

        new = np.newaxis
        self.d = 1.0/(self.fock_Od[:, new] - self.fock_Vd[new, :])

        self.D = 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :])

        ## Initial T1 amplitudes

        #self.T1 = self.fock_OV*self.d

        ## Initial T2 amplitudes

        #self.T2 = self.D*self.Voovv

        # Get MP2 energy

        self.update_energy()

        print('Energy guess:   {:<15.10f}'.format(self.Ecc + self.Ehf))

        # EXTERNAL CORRECTION

        ## Contribution of connected T3 terms to T2 equations. 

        self.T3onT2 = +0.5*np.einsum('mbfe, jimeaf -> ijab', (self.Vovvv - self.Vovvv.swapaxes(2,3)), self.T3) \
                      +    np.einsum('mbfe, ijmaef -> ijab', self.Vovvv, self.T3)                             \
                      -0.5*np.einsum('mnje, minbae -> ijab', self.Vooov, self.T3) \
                      +0.5*np.einsum('nmje, minbae -> ijab', self.Vooov, self.T3) \
                      -    np.einsum('mnje, imnabe -> ijab', self.Vooov, self.T3) \
                      +    np.einsum('me,   ijmabe -> ijab', self.fock_OV, self.T3)

        self.T3onT2 = self.T3onT2 + np.einsum('ijab -> jiba', self.T3onT2) 

        ## Contribution of connected T4 terms to T2 equations.
        
        self.T4onT2 = np.einsum('mnef, ijmnabef -> ijab', (self.Voovv - self.Voovv.swapaxes(2,3)), self.T4abaa)        

        self.T4onT2 += np.einsum('ijab -> jiba', self.T4onT2)

        self.T4onT2 = (1.0/4.0)*self.T4onT2 + np.einsum('mnef, ijmnabef -> ijab', self.Voovv, self.T4abab)

        # Setup iteration options
        self.rms1 = 0.0
        self.rms2 = 0.0
        max_rms = 1
        dE = 1
        ite = 1
        rms_LIM = 10**(-self.CC_CONV)
        E_LIM = 10**(-self.E_CONV)
        t0 = time.time()
        print('='*37)

        # Start CC iterations
        while abs(dE) > E_LIM or max_rms > rms_LIM:
            t = time.time()
            if ite > self.CC_MAXITER:
                raise NameError('CC equations did not converge')
            self.update_tau_and_te()
            self.update_Fint()
            self.update_Winf()        
            self.update_amp()
            dE = -self.Ecc
            self.update_energy()
            dE += self.Ecc
            max_rms = max(self.rms1, self.rms2)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {:< 15.10f}".format(self.Ecc))
            print("Energy change:         {:< 15.10f}".format(dE))
            print("Max RMS residue:       {:< 15.10f}".format(max_rms))
            print("Time required:         {:< 15.10f}".format(time.time() - t))
            print('='*37)
            ite += 1

        print('CC Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))
        print('CCSD iterations took %.2f seconds.\n' % (time.time() - t0))
        printmatrix(self.T1)
