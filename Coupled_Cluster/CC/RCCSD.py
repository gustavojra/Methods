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

    def V(self, arg):

        # Auxiliar method to handle different integrals type (i.e. virtual, occipied, alpha, beta)
        # This function is called as self.V(string)
        # where string contains four characters representing the four indices as
        # -> Capital letters for alpha orbitals
        # -> Lower case letters for beta orbitals 
        # -> ijklmn for occupied orbitals
        # -> abcdef for virtual orbitals
        
        def space(entry):

            # Auxiliar function to determine the orbital space of a given character

            if entry in 'ijklmn':
                return slice(0, self.ndocc)
            elif entry in 'IJKLMN':
                return slice(0, self.ndocc)
            elif entry in 'abcdef':
                return slice(self.ndocc, self.nmo)
            elif entry in 'ABCDEF':
                return slice(self.ndocc, self.nmo)

        x,y,z,w = arg

        # Case aaaa
        if arg.isupper():
            return self.Vanti[space(x), space(y), space(z), space(w)]

        # Case bbbb
        elif arg.islower():
            return self.Vanti[space(x), space(y), space(z), space(w)]

        # Case ab--
        elif x.isupper():
            # Case abab
            if z.isupper():
                return self.Vphys[space(x), space(y), space(z), space(w)]
            # Case abba
            if w.isupper():
                return -self.Vphys.transpose(0,1,3,2)[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

        # Case ba--
        elif y.isupper():
            # Case baab
            if z.isupper():
                return -self.Vphys.transpose(1,0,2,3)[space(x), space(y), space(z), space(w)]
            # Case baba
            if w.isupper():
                return self.Vphys.transpose(1,0,3,2)[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

    def update_energy(self):
        
        X = 2*self.T2 + 2*np.einsum('IA,JB->IJAB', self.T1, self.T1)
        X += - self.T2.transpose(1,0,2,3) - np.einsum('JA,IB->IJAB', self.T1, self.T1)
        self.Ecc = np.einsum('IjAb,IjAb->', X, self.V('IjAb'))

    def update_tau_and_te(self):

        self.tau = self.T2 + 0.5*np.einsum('IA,JB->IJAB', self.T1, self.T1)
        self.Te = self.T2 + np.einsum('IA,JB->IJAB', self.T1, self.T1)
    
    def update_Fint(self):

        # Slices
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nmo)

        # Update F(AE)
        self.Fae = np.zeros((self.nvir, self.nvir))
        self.Fae += self.fock_VV - 0.5*np.einsum('ME,MA->AE', self.fock_OV, self.T1)
        self.Fae += np.einsum('MF,MAFE->AE', self.T1, self.Vsa[o,v,v,v])
        self.Fae += -np.einsum('MnAf,MnEf->AE', self.tau, self.Vsa[o,o,v,v])

        # Update F(MI)
        self.Fmi = np.zeros((self.ndocc, self.ndocc))
        self.Fmi += self.fock_OO + 0.5*np.einsum('ME,IE->MI', self.fock_OV, self.T1)
        self.Fmi += np.einsum('NE,MNIE->MI', self.T1, self.Vsa[o,o,o,v])
        self.Fmi += np.einsum('INEF,MNEF->MI', self.tau, self.Vsa[o,o,v,v])

        # Update F(ME)
        self.Fme = np.zeros((self.ndocc, self.nvir))
        self.Fme += self.fock_OV + np.einsum('NF, MNEF-> ME', self.T1, self.Vsa[o,o,v,v])

    def update_Winf(self):

        # Slices
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nmo)

        # Clean up W cases
        self.W = {}

        # Update W(MnIj)
        self.Wmnij = np.zeros((self.ndocc, self.ndocc, self.ndocc, self.ndocc))
        self.Wmnij += self.Vphys[o,o,o,o]
        self.Wmnij += np.einsum('je, MnIe-> MnIj', self.T1, self.Vphys[o,o,o,v])
        self.Wmnij += np.einsum('IE,nMjE -> MnIj', self.T1, self.Vphys[o,o,o,v])
        self.Wmnij += (1.0/2.0)*np.einsum('IjEf,MnEf->MnIj', self.Te, self.Vphys[o,o,v,v])

        # Update W(AbEf)
        self.Wabef = np.zeros((self.nvir, self.nvir, self.nvir, self.nvir))
        self.Wabef += self.Vphys[v,v,v,v]
        self.Wabef += -np.einsum('mb, mAfE-> AbEf', self.T1, self.Vphys[o,v,v,v])
        self.Wabef += -np.einsum('MA, MbEf -> AbEf', self.T1, self.Vphys[o,v,v,v])
        self.Wabef += (1.0/2.0)*np.einsum('MnAb,MnEf->AbEf', self.Te, self.Vphys[o,o,v,v])

        # Update W(MBEJ)
        self.W.update({'MBEJ' : np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))})
        self.W['MBEJ'] += self.V('MBEJ')
        self.W['MBEJ'] += np.einsum('JF,MBEF->MBEJ', self.T1, self.V('MBEF'))
        self.W['MBEJ'] += -np.einsum('NB,MNEJ->MBEJ', self.T1, self.V('MNEJ'))
        X = 0.5*self.T2aa + np.einsum('JF,NB->JNFB', self.T1, self.T1)
        self.W['MBEJ'] += -np.einsum('JNFB,MNEF->MBEJ', X, self.V('MNEF'))
        self.W['MBEJ'] += 0.5*np.einsum('JnBf,MnEf->MBEJ', self.T2, self.V('MnEf'))

        # Update W(mbej)
        self.W.update({'mbej' : np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))})
        self.W['mbej'] += self.V('mbej')
        self.W['mbej'] += np.einsum('jf,mbef->mbej', self.T1, self.V('mbef'))
        self.W['mbej'] += -np.einsum('nb,mnej->mbej', self.T1, self.V('mnej'))
        X = 0.5*self.T2aa + np.einsum('jf,nb->jnfb', self.T1, self.T1)
        self.W['mbej'] += -np.einsum('jnfb,mnef->mbej', X, self.V('mnef'))
        self.W['mbej'] += 0.5*np.einsum('NjFb,mNeF->mbej', self.T2, self.V('mNeF'))

        # Update W(MbEj)
        self.W_MbEj = np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))
        self.W_MbEj += self.V('MbEj')
        self.W_MbEj += np.einsum('jf,MbEf->MbEj', self.T1, self.Vphys[o,v,v,v])
        self.W_MbEj += -np.einsum('nb,nMjE->MbEj', self.T1, self.Vphys[o,o,o,v])
        X = self.T2 + 2*np.einsum('jf,nb->jnfb', self.T1, self.T1)
        self.W_MbEj += -0.5*np.einsum('jnfb,MnEf->MbEj', X, self.Vphys[o,o,v,v])
        self.W_MbEj += 0.5*np.einsum('NjFb,MNEF->MbEj', self.T2, self.Vsa[o,o,v,v])

        # HERE    
    
        # Update W(MbeJ)
        self.W.update({'MbeJ' : np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))})
        self.W['MbeJ'] += self.V('MbeJ')
        self.W['MbeJ'] += np.einsum('JF,MbeF->MbeJ', self.T1, self.V('MbeF'))
        self.W['MbeJ'] += -np.einsum('nb,MneJ->MbeJ', self.T1, self.V('MneJ'))
        X = 0.5*self.T2 + np.einsum('JF,nb->JnFb', self.T1, self.T1)
        self.W['MbeJ'] += -np.einsum('JnFb,MneF->MbeJ', X, self.V('MneF'))

        # Update W(mBeJ)
        self.W.update({'mBeJ' : np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))})
        self.W['mBeJ'] += self.V('mBeJ')
        self.W['mBeJ'] += np.einsum('JF,mBeF->mBeJ', self.T1, self.V('mBeF'))
        self.W['mBeJ'] += -np.einsum('NB,mNeJ->mBeJ', self.T1, self.V('mNeJ'))
        X = 0.5*self.T2aa + np.einsum('JF,NB->JNFB', self.T1, self.T1)
        self.W['mBeJ'] += -np.einsum('JNFB,mNeF->mBeJ', X, self.V('mNeF'))
        self.W['mBeJ'] += 0.5*np.einsum('JnBf,mnef->mBeJ', self.T2, self.V('mnef'))

        # Update W(mBEj)
        self.W.update({'mBEj' : np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))})
        self.W['mBEj'] += self.V('mBEj')
        self.W['mBEj'] += np.einsum('jf,mBEf->mBEj', self.T1, self.V('mBEf'))
        self.W['mBEj'] += -np.einsum('NB,mNEj->mBEj', self.T1, self.V('mNEj'))
        X = 0.5*self.T2 + np.einsum('jf,NB->NjBf', self.T1, self.T1)
        self.W['mBEj'] += -np.einsum('NjBf,mNEf->mBEj', X, self.V('mNEf'))

    def update_amp(self):

        # Create a new set of amplitudes

        newT1 = np.zeros(self.T1.shape)

        newT2 = np.zeros(self.T2.shape)

        # Update T(IA)
        newT1 += self.fock_OV 
        newT1 += np.einsum('IE,AE->IA', self.T1, self.Fae)
        newT1 += -np.einsum('MA,MI->IA', self.T1, self.Fmi)
        newT1 += np.einsum('IMAE,ME->IA', self.T2aa, self.Fme)
        newT1 += np.einsum('ImAe,me->IA', self.T2, self.Fme)
        newT1 += np.einsum('ME,AMIE->IA', self.T1, self.V('AMIE'))
        newT1 += np.einsum('me,AmIe->IA', self.T1, self.V('AmIe'))
        newT1 += -0.5*np.einsum('MNAE,MNIE->IA', self.T2aa, self.V('MNIE'))
        newT1 += -np.einsum('MnAe,MnIe->IA', self.T2, self.V('MnIe'))
        newT1 += 0.5*np.einsum('IMEF,AMEF->IA', self.T2aa, self.V('AMEF'))
        newT1 += np.einsum('ImEf,AmEf->IA', self.T2, self.V('AmEf'))
        newT1 *= self.d

        # Update T(IjAb)

        newT2 += self.V('IjAb')
        X = self.Fae - 0.5*np.einsum('mb,me->be', self.T1, self.Fme)
        newT2 += np.einsum('IjAe,be->IjAb', self.T2, X)

        X = self.Fae - 0.5*np.einsum('MA,ME->AE', self.T1, self.Fme) 
        newT2 += np.einsum('IjEb,AE->IjAb', self.T2, X)

        X = self.Fmi + 0.5*np.einsum('je,me->mj', self.T1, self.Fme) 
        newT2 += -np.einsum('ImAb,mj->IjAb', self.T2, X)

        X = self.Fmi + 0.5*np.einsum('IE,ME->MI', self.T1, self.Fme) 
        newT2 += -np.einsum('MjAb,MI->IjAb', self.T2, X)

        X = self.T2 + np.einsum('MA,nb->MnAb', self.T1, self.T1)
        newT2 += np.einsum('MnAb,MnIj->IjAb', X, self.Wmnij)

        X = self.T2 + np.einsum('IE,jf->IjEf', self.T1, self.T1) 
        newT2 += np.einsum('IjEf,AbEf->IjAb', X, self.Wabef)

        newT2 += np.einsum('IMAE,MbEj->IjAb', self.T2aa, self.W_MbEj)
        newT2 += -np.einsum('IE,MA,MbEj->IjAb', self.T1, self.T1, self.V('MbEj'))

        newT2 += np.einsum('ImAe,mbej->IjAb', self.T2, self.W['mbej'])
        
        newT2 += np.einsum('ImEb,mAEj->IjAb', self.T2, self.W['mBEj'])
        newT2 += np.einsum('IE,mb,mAEj->IjAb', self.T1, self.T1, self.V('mAEj'))

        newT2 += np.einsum('MjAe,MbeI->IjAb', self.T2, self.W['MbeJ'])
        newT2 += np.einsum('je,MA,MbeI->IjAb', self.T1, self.T1, self.V('MbeI'))

        newT2 += np.einsum('jmbe,mAeI->IjAb', self.T2aa, self.W['mBeJ'])
        newT2 += -np.einsum('je,mb,mAeI->IjAb', self.T1, self.T1, self.V('mAeI'))

        newT2 += np.einsum('MjEb,MAEI->IjAb', self.T2, self.W['MBEJ'])

        newT2 += np.einsum('IE,AbEj->IjAb', self.T1, self.V('AbEj'))
        newT2 += -np.einsum('je,AbeI->IjAb', self.T1, self.V('AbeI'))
        newT2 += -np.einsum('MA,MbIj->IjAb', self.T1, self.V('MbIj'))
        newT2 += np.einsum('mb,mAIj->IjAb', self.T1, self.V('mAIj'))

        newT2 *= self.D

        # Compute RMS

        self.rms1 = np.sqrt(np.sum(np.square(newT1 - self.T1 )))/(self.ndocc*self.nvir)
        self.rms2 = np.sqrt(np.sum(np.square(newT2 - self.T2 )))/(self.ndocc*self.ndocc*self.nvir*self.nvir)

        # Save new amplitudes

        self.T1 = newT1
        self.T2 = newT2
        self.T2aa = self.T2 - self.T2.transpose(1,0,2,3)

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

        # Save reference wavefunction properties
        self.Ehf = wfn.energy() 
        self.nmo = wfn.nmo()
        self.nelec = wfn.nalpha() + wfn.nbeta()
        if self.nelec % 2 != 0:
            NameError('Invalid number of electrons for RHF') 
        self.ndocc = int(self.nelec/2)
        self.nvir = self.nmo - self.ndocc
        self.C = wfn.Ca()
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()

        # Save Options
        self.CC_CONV = CC_CONV
        self.E_CONV = E_CONV
        self.CC_MAXITER = CC_MAXITER

        print("Number of electrons:              {}".format(self.nelec))
        print("Number of Doubly Occupied MOs:    {}".format(self.ndocc))
        print("Number of MOs:                    {}".format(self.nmo))

        print("\n Transforming integrals...")

        mints = psi4.core.MintsHelper(wfn.basisset())
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

        # Save two-electron integral in physicists notation (3 Spin cases)
        self.Vanti = Vchem.swapaxes(1,2)
        self.Vanti = self.Vanti - self.Vanti.swapaxes(2,3)

        self.Vphys = Vchem.swapaxes(1,2)
        self.Vsa = 2*self.Vphys - self.Vphys.swapaxes(2,3)

        self.compute()

    def compute(self):

        # Auxiliar D matrices

        new = np.newaxis
        self.d = 1.0/(self.fock_Od[:, new] - self.fock_Vd[new, :])

        self.D = 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :])

        # Initial T1 amplitudes

        self.T1 = self.fock_OV*self.d

        # Initial T2 amplitudes

        self.T2 = self.D*self.V('IjAb')
        self.T2aa = self.T2 - self.T2.transpose(1,0,2,3)

        # Get MP2 energy

        self.update_energy()

        print('MP2 Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))

        self.rms1 = 0.0

        self.rms2 = 0.0

        max_rms = 1
        dE = 1
        ite = 1

        rms_LIM = 10**(-self.CC_CONV)
        E_LIM = 10**(-self.E_CONV)
        t0 = time.time()
        print('='*37)
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
