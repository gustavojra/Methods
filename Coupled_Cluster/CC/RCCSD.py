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
            return (self.Vint - self.Vint.swapaxes(2,3))[space(x), space(y), space(z), space(w)]

        # Case bbbb
        elif arg.islower():
            return (self.Vint - self.Vint.swapaxes(2,3))[space(x), space(y), space(z), space(w)]

        # Case ab--
        elif x.isupper():
            # Case abab
            if z.isupper():
                return self.Vint[space(x), space(y), space(z), space(w)]
            # Case abba
            if w.isupper():
                return -(self.Vint[space(x), space(y), space(z), space(w)])
            else:
                raise NameError('Invalid integral key')

        # Case ba--
        elif y.isupper():
            # Case baab
            if z.isupper():
                return -(self.Vint[space(x), space(y), space(z), space(w)])
            # Case baba
            if w.isupper():
                return self.Vint[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

    def update_energy(self):
        
        self.Ecc  = 2*np.einsum('IA,IA->', self.fock_OV, self.T1)
    
        X = self.T2aa + 2*np.einsum('IA,JB->IJAB', self.T1, self.T1)
        self.Ecc += (1.0/4.0)*np.einsum('IJAB,IJAB->', X, self.V('IJAB'))
    
        X = self.T2aa + 2*np.einsum('ia,jb->ijab', self.T1, self.T1)
        self.Ecc += (1.0/4.0)*np.einsum('ijab,ijab->', X, self.V('ijab'))
    
        X = self.T2 + np.einsum('IA,jb->IjAb', self.T1, self.T1)
        self.Ecc += np.einsum('IjAb,IjAb->', X, self.V('IjAb'))
    
    
    def update_Fint(self):

        # Clean up F cases
        self.F = {}

        # Update F(AE)
        self.F.update({'AE' : np.zeros((self.nvir, self.nvir))})
        self.F['AE'] += self.fock_VV - 0.5*np.einsum('ME,MA->AE', self.fock_OV, self.T1)
        self.F['AE'] += np.einsum('MF,AMEF->AE', self.T1, self.V('AMEF'))
        self.F['AE'] += np.einsum('mf,AmEf->AE', self.T1, self.V('AmEf'))

        X = self.T2aa + 0.5*np.einsum('MA,NF->MNAF', self.T1, self.T1) - 0.5*np.einsum('MF,NA->MNAF', self.T1, self.T1)
        self.F['AE'] += -0.5*np.einsum('MNAF,MNEF->AE', X, self.V('MNEF'))

        X = self.T2 + 0.5*np.einsum('MA,nf->MnAf', self.T1, self.T1)
        self.F['AE'] += -np.einsum('MnAf,MnEf->AE', X, self.V('MnEf'))

        # Update F(ae)
        self.F.update({'ae' : np.zeros((self.nvir, self.nvir))})
        self.F['ae'] += self.fock_VV - 0.5*np.einsum('me,ma->ae', self.fock_OV, self.T1)
        self.F['ae'] += np.einsum('mf,amef->ae', self.T1, self.V('amef'))
        self.F['ae'] += np.einsum('MF,aMeF->ae', self.T1, self.V('aMeF'))

        X = self.T2aa + 0.5*np.einsum('ma,nf->mnaf', self.T1, self.T1) - 0.5*np.einsum('mf,na->mnaf', self.T1, self.T1)
        self.F['ae'] += -0.5*np.einsum('mnaf,mnef->ae', X, self.V('mnef'))

        X = self.T2 + 0.5*np.einsum('ma,NF->NmFa', self.T1, self.T1)
        self.F['ae'] += -np.einsum('NmFa,mNeF->ae', X, self.V('mNeF'))

        # Update F(MI)
        self.F.update({'MI' : np.zeros((self.ndocc, self.ndocc))})
        self.F['MI'] += self.fock_OO + 0.5*np.einsum('ME,IE->MI', self.fock_OV, self.T1)
        self.F['MI'] += np.einsum('NE,MNIE->MI', self.T1, self.V('MNIE'))
        self.F['MI'] += np.einsum('ne,MnIe->MI', self.T1, self.V('MnIe'))

        X = self.T2aa + 0.5*np.einsum('IE,NF->INEF', self.T1, self.T1) - 0.5*np.einsum('IF,NE->INEF', self.T1, self.T1)
        self.F['MI'] += +0.5*np.einsum('INEF,MNEF->MI', X, self.V('MNEF'))

        X = self.T2 + 0.5*np.einsum('IE,nf->InEf', self.T1, self.T1)
        self.F['MI'] += np.einsum('InEf,MnEf->MI', X, self.V('MnEf'))

        # Update F(mi)
        self.F.update({'mi' : np.zeros((self.ndocc, self.ndocc))})
        self.F['mi'] += self.fock_OO + 0.5*np.einsum('me,ie->mi', self.fock_OV, self.T1)
        self.F['mi'] += np.einsum('ne,mnie->mi', self.T1, self.V('mnie'))
        self.F['mi'] += np.einsum('NE,mNiE->mi', self.T1, self.V('mNiE'))

        X = self.T2aa + 0.5*np.einsum('ie,nf->inef', self.T1, self.T1) - 0.5*np.einsum('if,ne->inef', self.T1, self.T1)
        self.F['mi'] += +0.5*np.einsum('inef,mnef->mi', X, self.V('mnef'))

        X = self.T2 + 0.5*np.einsum('ie,NF->NiFe', self.T1, self.T1)
        self.F['mi'] += np.einsum('NiFe,mNeF->mi', X, self.V('mNeF'))

        # Update F(ME)
        self.F.update({'ME' : np.zeros((self.ndocc, self.nvir))})
        self.F['ME'] += self.fock_OV + np.einsum('NF, MNEF-> ME', self.T1, self.V('MNEF')) + np.einsum('nf, MnEf-> ME', self.T1, self.V('MnEf'))

        # Update F(me)
        self.F.update({'me' : np.zeros((self.ndocc, self.nvir))})
        self.F['me'] += self.fock_OV + np.einsum('nf, mnef-> me', self.T1, self.V('mnef')) + np.einsum('NF, mNeF-> me', self.T1, self.V('mNeF'))

    def update_Winf(self):

        # Clean up W cases
        self.W = {}

        # Update W(MNIJ)
        self.W.update({'MNIJ' : np.zeros((self.ndocc, self.ndocc, self.ndocc, self.ndocc))})
        self.W['MNIJ'] += self.V('MNIJ')
        self.W['MNIJ'] += np.einsum('JE, MNIE-> MNIJ', self.T1, self.V('MNIE'))
        self.W['MNIJ'] += -np.einsum('IE, MNJE-> MNIJ', self.T1, self.V('MNJE'))
        X = self.T2aa + np.einsum('IE,JF->IJEF', self.T1, self.T1) - np.einsum('IF,JE->IJEF', self.T1, self.T1)
        self.W['MNIJ'] += (1.0/4.0)*np.einsum('IJEF,MNEF->MNIJ', X, self.V('MNEF'))

        # Update W(mnij)
        self.W.update({'mnij' : np.zeros((self.ndocc, self.ndocc, self.ndocc, self.ndocc))})
        self.W['mnij'] += self.V('mnij')
        self.W['mnij'] += np.einsum('je, mnie-> mnij', self.T1, self.V('mnie'))
        self.W['mnij'] += -np.einsum('ie, mnje-> mnij', self.T1, self.V('mnje'))
        X = self.T2aa + np.einsum('ie,jf->ijef', self.T1, self.T1) - np.einsum('if,je->ijef', self.T1, self.T1)
        self.W['mnij'] += (1.0/4.0)*np.einsum('ijef,mnef->mnij', X, self.V('mnef'))

        # Update W(MnIj)
        self.W.update({'MnIj' : np.zeros((self.ndocc, self.ndocc, self.ndocc, self.ndocc))})
        self.W['MnIj'] += self.V('MnIj')
        self.W['MnIj'] += np.einsum('je, MnIe-> MnIj', self.T1, self.V('MnIe'))
        self.W['MnIj'] += -np.einsum('IE,MnjE -> MnIj', self.T1, self.V('MnjE'))
        X = self.T2 + np.einsum('IE,jf->IjEf', self.T1, self.T1) 
        self.W['MnIj'] += (1.0/2.0)*np.einsum('IjEf,MnEf->MnIj', X, self.V('MnEf'))

        # Update W(ABEF)
        self.W.update({'ABEF' : np.zeros((self.nvir, self.nvir, self.nvir, self.nvir))})
        self.W['ABEF'] += self.V('ABEF')
        self.W['ABEF'] += -np.einsum('MB, AMEF-> ABEF', self.T1, self.V('AMEF'))
        self.W['ABEF'] += np.einsum('MA, BMEF-> ABEF', self.T1, self.V('BMEF'))
        X = self.T2aa + np.einsum('MA,NB->MNAB', self.T1, self.T1) - np.einsum('MB,NA->MNAB', self.T1, self.T1)
        self.W['ABEF'] += (1.0/4.0)*np.einsum('MNAB,MNEF->ABEF', X, self.V('MNEF'))

        # Update W(abef)
        self.W.update({'abef' : np.zeros((self.nvir, self.nvir, self.nvir, self.nvir))})
        self.W['abef'] += self.V('abef')
        self.W['abef'] += -np.einsum('mb, amef-> abef', self.T1, self.V('amef'))
        self.W['abef'] += np.einsum('ma, bmef-> abef', self.T1, self.V('bmef'))
        X = self.T2aa + np.einsum('ma,nb->mnab', self.T1, self.T1) - np.einsum('mb,na->mnab', self.T1, self.T1)
        self.W['abef'] += (1.0/4.0)*np.einsum('mnab,mnef->abef', X, self.V('mnef'))

        # Update W(AbEf)
        self.W.update({'AbEf' : np.zeros((self.nvir, self.nvir, self.nvir, self.nvir))})
        self.W['AbEf'] += self.V('AbEf')
        self.W['AbEf'] += -np.einsum('mb, AmEf-> AbEf', self.T1, self.V('AmEf'))
        self.W['AbEf'] += np.einsum('MA,bMEf -> AbEf', self.T1, self.V('bMEf'))
        X = self.T2 + np.einsum('MA,nb->MnAb', self.T1, self.T1) 
        self.W['AbEf'] += (1.0/2.0)*np.einsum('MnAb,MnEf->AbEf', X, self.V('MnEf'))

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
        self.W.update({'MbEj' : np.zeros((self.ndocc, self.nvir, self.nvir, self.ndocc))})
        self.W['MbEj'] += self.V('MbEj')
        self.W['MbEj'] += np.einsum('jf,MbEf->MbEj', self.T1, self.V('MbEf'))
        self.W['MbEj'] += -np.einsum('nb,MnEj->MbEj', self.T1, self.V('MnEj'))
        X = 0.5*self.T2aa + np.einsum('jf,nb->jnfb', self.T1, self.T1)
        self.W['MbEj'] += -np.einsum('jnfb,MnEf->MbEj', X, self.V('MnEf'))
        self.W['MbEj'] += 0.5*np.einsum('NjFb,MNEF->MbEj', self.T2, self.V('MNEF'))

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
        newT1 += np.einsum('IE,AE->IA', self.T1, self.F['AE'])
        newT1 += -np.einsum('MA,MI->IA', self.T1, self.F['MI'])
        newT1 += np.einsum('IMAE,ME->IA', self.T2aa, self.F['ME'])
        newT1 += np.einsum('ImAe,me->IA', self.T2, self.F['me'])
        newT1 += np.einsum('ME,AMIE->IA', self.T1, self.V('AMIE'))
        newT1 += np.einsum('me,AmIe->IA', self.T1, self.V('AmIe'))
        newT1 += -0.5*np.einsum('MNAE,MNIE->IA', self.T2aa, self.V('MNIE'))
        newT1 += -np.einsum('MnAe,MnIe->IA', self.T2, self.V('MnIe'))
        newT1 += 0.5*np.einsum('IMEF,AMEF->IA', self.T2aa, self.V('AMEF'))
        newT1 += np.einsum('ImEf,AmEf->IA', self.T2, self.V('AmEf'))
        newT1 *= self.d

        # Update T(IjAb)

        newT2 += self.V('IjAb')
        X = self.F['ae'] - 0.5*np.einsum('mb,me->be', self.T1, self.F['me'])
        newT2 += np.einsum('IjAe,be->IjAb', self.T2, X)

        X = self.F['AE'] - 0.5*np.einsum('MA,ME->AE', self.T1, self.F['ME']) 
        newT2 += np.einsum('IjEb,AE->IjAb', self.T2, X)

        X = self.F['mi'] + 0.5*np.einsum('je,me->mj', self.T1, self.F['me']) 
        newT2 += -np.einsum('ImAb,mj->IjAb', self.T2, X)

        X = self.F['MI'] + 0.5*np.einsum('IE,ME->MI', self.T1, self.F['ME']) 
        newT2 += -np.einsum('MjAb,MI->IjAb', self.T2, X)

        X = self.T2 + np.einsum('MA,nb->MnAb', self.T1, self.T1)
        newT2 += np.einsum('MnAb,MnIj->IjAb', X, self.W['MnIj'])

        X = self.T2 + np.einsum('IE,jf->IjEf', self.T1, self.T1) 
        newT2 += np.einsum('IjEf,AbEf->IjAb', X, self.W['AbEf'])

        newT2 += np.einsum('IMAE,MbEj->IjAb', self.T2aa, self.W['MbEj'])
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

        self.rms1 = np.sqrt(np.sum(np.square(newT1 - self.T1)))/(self.ndocc*self.nvir)
        self.rms2 = np.sqrt(np.sum(np.square(newT2 - self.T2)))/(self.ndocc*self.nvir)**2

        # Save new amplitudes

        self.T1 = newT1
        self.T2 = newT2
        self.T2aa = self.T2 - self.T2.swapaxes(2,3)

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

        # Save reference wavefunction properties
        self.Ehf = wfn.energy() 
        self.nmo = wfn.nmo()
        self.nelec = wfn.nalpha() + wfn.nbeta()
        if self.nelec % 2 != 0:
            raise NameError('Odd number of electron incompatible with RHF')
        self.ndocc = int(self.nelec/2)
        self.nvir = self.nmo - self.ndocc
        self.C = wfn.Ca()
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()

        # Save Options
        self.CC_CONV = CC_CONV
        self.E_CONV = E_CONV
        self.CC_MAXITER = CC_MAXITER

        print("Number of electrons:            {}".format(self.nelec))
        print("Number of Doubly Occupied MOs   {}".format(self.ndocc))
        print("Number of MOs:                  {}".format(self.nmo))

        print("\n Transforming integrals...")

        mints = psi4.core.MintsHelper(wfn.basisset())
        self.mints = mints
        # One electron integral
        h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        h = np.einsum('up,vq,uv->pq', self.C, self.C, h)

        V = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
    
        # Slices
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nmo)

        # Form the full fock matrices
        f = h + 2*np.einsum('pqkk->pq', V[:,:,o,o]) - np.einsum('pkqk->pq', V[:,o,:,o])

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
        self.Vint = V.swapaxes(1,2)
        #self.Vint = self.Vint - self.Vint.swapaxes(2,3)

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
        self.T2aa = self.T2 - self.T2.swapaxes(2,3)

        # Get MP2 energy

        self.update_energy()

        print('MP2 Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))

        self.rms1 = 0.0

        # Initial T2 amplitudes

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
