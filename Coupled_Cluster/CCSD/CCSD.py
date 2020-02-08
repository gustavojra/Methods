import psi4
import os
import sys
import numpy as np
import time
import copy

sys.path.append('../../Aux')
from tools import *

np.set_printoptions(suppress=True, linewidth=120)

class CCSD:

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
                return slice(0, self.nbeta)
            elif entry in 'IJKLMN':
                return slice(0, self.nalpha)
            elif entry in 'abcdef':
                return slice(self.nbeta, self.nmo)
            elif entry in 'ABCDEF':
                return slice(self.nalpha, self.nmo)

        x,y,z,w = arg

        # Case aaaa
        if arg.isupper():
            return self.Vaaaa[space(x), space(y), space(z), space(w)]

        # Case bbbb
        elif arg.islower():
            return self.Vbbbb[space(x), space(y), space(z), space(w)]

        # Case ab--
        elif x.isupper():
            # Case abab
            if z.isupper():
                return self.Vabab[space(x), space(y), space(z), space(w)]
            # Case abba
            if w.isupper():
                return self.Vabba[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

        # Case ba--
        elif y.isupper():
            # Case baab
            if z.isupper():
                return self.Vbaab[space(x), space(y), space(z), space(w)]
            # Case baba
            if w.isupper():
                return self.Vbaba[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

    def update_energy(self):
        
        self.Ecc  = np.einsum('IA,IA->', self.fock_OV, self.T1['IA'])
        self.Ecc += np.einsum('ia,ia->', self.fock_ov, self.T1['ia'])
    
        X = self.T2['IJAB'] + 2*np.einsum('IA,JB->IJAB', self.T1['IA'], self.T1['IA'])
        self.Ecc += (1.0/4.0)*np.einsum('IJAB,IJAB->', X, self.V('IJAB'))
    
        X = self.T2['ijab'] + 2*np.einsum('ia,jb->ijab', self.T1['ia'], self.T1['ia'])
        self.Ecc += (1.0/4.0)*np.einsum('ijab,ijab->', X, self.V('ijab'))
    
        X = self.T2['IjAb'] + np.einsum('IA,jb->IjAb', self.T1['IA'], self.T1['ia'])
        self.Ecc += np.einsum('IjAb,IjAb->', X, self.V('IjAb'))
    
    
    def update_Fint(self):

        # Clean up F cases
        self.F = {}

        # Update F(AE)
        self.F.update({'AE' : np.zeros((self.avir, self.avir))})
        self.F['AE'] += self.fock_VV - 0.5*np.einsum('ME,MA->AE', self.fock_OV, self.T1['IA'])
        self.F['AE'] += np.einsum('MF,AMEF->AE', self.T1['IA'], self.V('AMEF'))
        self.F['AE'] += np.einsum('mf,AmEf->AE', self.T1['ia'], self.V('AmEf'))

        X = self.T2['IJAB'] + 0.5*np.einsum('MA,NF->MNAF', self.T1['IA'], self.T1['IA']) - 0.5*np.einsum('MF,NA->MNAF', self.T1['IA'], self.T1['IA'])
        self.F['AE'] += -0.5*np.einsum('MNAF,MNEF->AE', X, self.V('MNEF'))

        X = self.T2['IjAb'] + 0.5*np.einsum('MA,nf->MnAf', self.T1['IA'], self.T1['ia'])
        self.F['AE'] += -np.einsum('MnAf,MnEf->AE', X, self.V('MnEf'))

        # Update F(ae)
        self.F.update({'ae' : np.zeros((self.bvir, self.bvir))})
        self.F['ae'] += self.fock_vv - 0.5*np.einsum('me,ma->ae', self.fock_ov, self.T1['ia'])
        self.F['ae'] += np.einsum('mf,amef->ae', self.T1['ia'], self.V('amef'))
        self.F['ae'] += np.einsum('MF,aMeF->ae', self.T1['IA'], self.V('aMeF'))

        X = self.T2['ijab'] + 0.5*np.einsum('ma,nf->mnaf', self.T1['ia'], self.T1['ia']) - 0.5*np.einsum('mf,na->mnaf', self.T1['ia'], self.T1['ia'])
        self.F['ae'] += -0.5*np.einsum('mnaf,mnef->ae', X, self.V('mnef'))

        X = self.T2['IjAb'] + 0.5*np.einsum('ma,NF->NmFa', self.T1['ia'], self.T1['IA'])
        self.F['ae'] += -np.einsum('NmFa,mNeF->ae', X, self.V('mNeF'))

        # Update F(MI)
        self.F.update({'MI' : np.zeros((self.nalpha, self.nalpha))})
        self.F['MI'] += self.fock_OO + 0.5*np.einsum('ME,IE->MI', self.fock_OV, self.T1['IA'])
        self.F['MI'] += np.einsum('NE,MNIE->MI', self.T1['IA'], self.V('MNIE'))
        self.F['MI'] += np.einsum('ne,MnIe->MI', self.T1['ia'], self.V('MnIe'))

        X = self.T2['IJAB'] + 0.5*np.einsum('IE,NF->INEF', self.T1['IA'], self.T1['IA']) - 0.5*np.einsum('IF,NE->INEF', self.T1['IA'], self.T1['IA'])
        self.F['MI'] += +0.5*np.einsum('INEF,MNEF->MI', X, self.V('MNEF'))

        X = self.T2['IjAb'] + 0.5*np.einsum('IE,nf->InEf', self.T1['IA'], self.T1['ia'])
        self.F['MI'] += np.einsum('InEf,MnEf->MI', X, self.V('MnEf'))

        # Update F(mi)
        self.F.update({'mi' : np.zeros((self.nbeta, self.nbeta))})
        self.F['mi'] += self.fock_oo + 0.5*np.einsum('me,ie->mi', self.fock_ov, self.T1['ia'])
        self.F['mi'] += np.einsum('ne,mnie->mi', self.T1['ia'], self.V('mnie'))
        self.F['mi'] += np.einsum('NE,mNiE->mi', self.T1['IA'], self.V('mNiE'))

        X = self.T2['ijab'] + 0.5*np.einsum('ie,nf->inef', self.T1['ia'], self.T1['ia']) - 0.5*np.einsum('if,ne->inef', self.T1['ia'], self.T1['ia'])
        self.F['mi'] += +0.5*np.einsum('inef,mnef->mi', X, self.V('mnef'))

        X = self.T2['IjAb'] + 0.5*np.einsum('ie,NF->NiFe', self.T1['ia'], self.T1['IA'])
        self.F['mi'] += np.einsum('NiFe,mNeF->mi', X, self.V('mNeF'))

        # Update F(ME)
        self.F.update({'ME' : np.zeros((self.nalpha, self.avir))})
        self.F['ME'] += self.fock_OV + np.einsum('NF, MNEF-> ME', self.T1['IA'], self.V('MNEF')) + np.einsum('nf, MnEf-> ME', self.T1['ia'], self.V('MnEf'))

        # Update F(me)
        self.F.update({'me' : np.zeros((self.nbeta, self.bvir))})
        self.F['me'] += self.fock_ov + np.einsum('nf, mnef-> me', self.T1['ia'], self.V('mnef')) + np.einsum('NF, mNeF-> me', self.T1['IA'], self.V('mNeF'))

    def update_Winf(self):

        # Clean up W cases
        self.W = {}

        # Update W(MNIJ)
        self.W.update({'MNIJ' : np.zeros((self.nalpha, self.nalpha, self.nalpha, self.nalpha))})
        self.W['MNIJ'] += self.V('MNIJ')
        self.W['MNIJ'] += np.einsum('JE, MNIE-> MNIJ', self.T1['IA'], self.V('MNIE'))
        self.W['MNIJ'] += -np.einsum('IE, MNJE-> MNIJ', self.T1['IA'], self.V('MNJE'))
        X = self.T2['IJAB'] + np.einsum('IE,JF->IJEF', self.T1['IA'], self.T1['IA']) - np.einsum('IF,JE->IJEF', self.T1['IA'], self.T1['IA'])
        self.W['MNIJ'] += (1.0/4.0)*np.einsum('IJEF,MNEF->MNIJ', X, self.V('MNEF'))

        # Update W(mnij)
        self.W.update({'mnij' : np.zeros((self.nbeta, self.nbeta, self.nbeta, self.nbeta))})
        self.W['mnij'] += self.V('mnij')
        self.W['mnij'] += np.einsum('je, mnie-> mnij', self.T1['ia'], self.V('mnie'))
        self.W['mnij'] += -np.einsum('ie, mnje-> mnij', self.T1['ia'], self.V('mnje'))
        X = self.T2['ijab'] + np.einsum('ie,jf->ijef', self.T1['ia'], self.T1['ia']) - np.einsum('if,je->ijef', self.T1['ia'], self.T1['ia'])
        self.W['mnij'] += (1.0/4.0)*np.einsum('ijef,mnef->mnij', X, self.V('mnef'))

        # Update W(MnIj)
        self.W.update({'MnIj' : np.zeros((self.nalpha, self.nbeta, self.nalpha, self.nbeta))})
        self.W['MnIj'] += self.V('MnIj')
        self.W['MnIj'] += np.einsum('je, MnIe-> MnIj', self.T1['ia'], self.V('MnIe'))
        self.W['MnIj'] += -np.einsum('IE,MnjE -> MnIj', self.T1['IA'], self.V('MnjE'))
        X = self.T2['IjAb'] + np.einsum('IE,jf->IjEf', self.T1['IA'], self.T1['ia']) 
        self.W['MnIj'] += (1.0/2.0)*np.einsum('IjEf,MnEf->MnIj', X, self.V('MnEf'))

        # Update W(ABEF)
        self.W.update({'ABEF' : np.zeros((self.avir, self.avir, self.avir, self.avir))})
        self.W['ABEF'] += self.V('ABEF')
        self.W['ABEF'] += -np.einsum('MB, AMEF-> ABEF', self.T1['IA'], self.V('AMEF'))
        self.W['ABEF'] += np.einsum('MA, BMEF-> ABEF', self.T1['IA'], self.V('BMEF'))
        X = self.T2['IJAB'] + np.einsum('MA,NB->MNAB', self.T1['IA'], self.T1['IA']) - np.einsum('MB,NA->MNAB', self.T1['IA'], self.T1['IA'])
        self.W['ABEF'] += (1.0/4.0)*np.einsum('MNAB,MNEF->ABEF', X, self.V('MNEF'))

        # Update W(abef)
        self.W.update({'abef' : np.zeros((self.bvir, self.bvir, self.bvir, self.bvir))})
        self.W['abef'] += self.V('abef')
        self.W['abef'] += -np.einsum('mb, amef-> abef', self.T1['ia'], self.V('amef'))
        self.W['abef'] += np.einsum('ma, bmef-> abef', self.T1['ia'], self.V('bmef'))
        X = self.T2['ijab'] + np.einsum('ma,nb->mnab', self.T1['ia'], self.T1['ia']) - np.einsum('mb,na->mnab', self.T1['ia'], self.T1['ia'])
        self.W['abef'] += (1.0/4.0)*np.einsum('mnab,mnef->abef', X, self.V('mnef'))

        # Update W(AbEf)
        self.W.update({'AbEf' : np.zeros((self.avir, self.bvir, self.avir, self.bvir))})
        self.W['AbEf'] += self.V('AbEf')
        self.W['AbEf'] += -np.einsum('mb, AmEf-> AbEf', self.T1['ia'], self.V('AmEf'))
        self.W['AbEf'] += np.einsum('MA,bMEf -> AbEf', self.T1['IA'], self.V('bMEf'))
        X = self.T2['IjAb'] + np.einsum('MA,nb->MnAb', self.T1['IA'], self.T1['ia']) 
        self.W['AbEf'] += (1.0/2.0)*np.einsum('MnAb,MnEf->AbEf', X, self.V('MnEf'))

        # Update W(MBEJ)
        self.W.update({'MBEJ' : np.zeros((self.nalpha, self.avir, self.avir, self.nalpha))})
        self.W['MBEJ'] += self.V('MBEJ')
        self.W['MBEJ'] += np.einsum('JF,MBEF->MBEJ', self.T1['IA'], self.V('MBEF'))
        self.W['MBEJ'] += -np.einsum('NB,MNEJ->MBEJ', self.T1['IA'], self.V('MNEJ'))
        X = 0.5*self.T2['IJAB'] + np.einsum('JF,NB->JNFB', self.T1['IA'], self.T1['IA'])
        self.W['MBEJ'] += -np.einsum('JNFB,MNEF->MBEJ', X, self.V('MNEF'))
        self.W['MBEJ'] += 0.5*np.einsum('JnBf,MnEf->MBEJ', self.T2['IjAb'], self.V('MnEf'))

        # Update W(mbej)
        self.W.update({'mbej' : np.zeros((self.nbeta, self.bvir, self.bvir, self.nbeta))})
        self.W['mbej'] += self.V('mbej')
        self.W['mbej'] += np.einsum('jf,mbef->mbej', self.T1['ia'], self.V('mbef'))
        self.W['mbej'] += -np.einsum('nb,mnej->mbej', self.T1['ia'], self.V('mnej'))
        X = 0.5*self.T2['ijab'] + np.einsum('jf,nb->jnfb', self.T1['ia'], self.T1['ia'])
        self.W['mbej'] += -np.einsum('jnfb,mnef->mbej', X, self.V('mnef'))
        self.W['mbej'] += 0.5*np.einsum('NjFb,mNeF->mbej', self.T2['IjAb'], self.V('mNeF'))

        # Update W(MbEj)
        self.W.update({'MbEj' : np.zeros((self.nalpha, self.bvir, self.avir, self.nbeta))})
        self.W['MbEj'] += self.V('MbEj')
        self.W['MbEj'] += np.einsum('jf,MbEf->MbEj', self.T1['ia'], self.V('MbEf'))
        self.W['MbEj'] += -np.einsum('nb,MnEj->MbEj', self.T1['ia'], self.V('MnEj'))
        X = 0.5*self.T2['ijab'] + np.einsum('jf,nb->jnfb', self.T1['ia'], self.T1['ia'])
        self.W['MbEj'] += -np.einsum('jnfb,MnEf->MbEj', X, self.V('MnEf'))
        self.W['MbEj'] += 0.5*np.einsum('NjFb,MNEF->MbEj', self.T2['IjAb'], self.V('MNEF'))

        # Update W(MbeJ)
        self.W.update({'MbeJ' : np.zeros((self.nalpha, self.bvir, self.bvir, self.nalpha))})
        self.W['MbeJ'] += self.V('MbeJ')
        self.W['MbeJ'] += np.einsum('JF,MbeF->MbeJ', self.T1['IA'], self.V('MbeF'))
        self.W['MbeJ'] += -np.einsum('nb,MneJ->MbeJ', self.T1['ia'], self.V('MneJ'))
        X = 0.5*self.T2['IjAb'] + np.einsum('JF,nb->JnFb', self.T1['IA'], self.T1['ia'])
        self.W['MbeJ'] += -np.einsum('JnFb,MneF->MbeJ', X, self.V('MneF'))

        # Update W(mBeJ)
        self.W.update({'mBeJ' : np.zeros((self.nbeta, self.avir, self.bvir, self.nalpha))})
        self.W['mBeJ'] += self.V('mBeJ')
        self.W['mBeJ'] += np.einsum('JF,mBeF->mBeJ', self.T1['IA'], self.V('mBeF'))
        self.W['mBeJ'] += -np.einsum('NB,mNeJ->mBeJ', self.T1['IA'], self.V('mNeJ'))
        X = 0.5*self.T2['IJAB'] + np.einsum('JF,NB->JNFB', self.T1['IA'], self.T1['IA'])
        self.W['mBeJ'] += -np.einsum('JNFB,mNeF->mBeJ', X, self.V('mNeF'))
        self.W['mBeJ'] += 0.5*np.einsum('JnBf,mnef->mBeJ', self.T2['IjAb'], self.V('mnef'))

        # Update W(mBEj)
        self.W.update({'mBEj' : np.zeros((self.nbeta, self.avir, self.avir, self.nbeta))})
        self.W['mBEj'] += self.V('mBEj')
        self.W['mBEj'] += np.einsum('jf,mBEf->mBEj', self.T1['ia'], self.V('mBEf'))
        self.W['mBEj'] += -np.einsum('NB,mNEj->mBEj', self.T1['IA'], self.V('mNEj'))
        X = 0.5*self.T2['IjAb'] + np.einsum('jf,NB->NjBf', self.T1['ia'], self.T1['IA'])
        self.W['mBEj'] += -np.einsum('NjBf,mNEf->mBEj', X, self.V('mNEf'))

    def update_amp(self):

        # Create a new set of amplitudes

        newT1 = {
        'IA' : np.zeros(self.T1['IA'].shape),
        'ia' : np.zeros(self.T1['ia'].shape)
        }

        newT2 = {
        'IJAB' : np.zeros(self.T2['IJAB'].shape),
        'ijab' : np.zeros(self.T2['ijab'].shape),
        'IjAb' : np.zeros(self.T2['IjAb'].shape),
        }

        # Update T(IA)
        newT1['IA'] += self.fock_OV 
        newT1['IA'] += np.einsum('IE,AE->IA', self.T1['IA'], self.F['AE'])
        newT1['IA'] += -np.einsum('MA,MI->IA', self.T1['IA'], self.F['MI'])
        newT1['IA'] += np.einsum('IMAE,ME->IA', self.T2['IJAB'], self.F['ME'])
        newT1['IA'] += np.einsum('ImAe,me->IA', self.T2['IjAb'], self.F['me'])
        newT1['IA'] += np.einsum('ME,AMIE->IA', self.T1['IA'], self.V('AMIE'))
        newT1['IA'] += np.einsum('me,AmIe->IA', self.T1['ia'], self.V('AmIe'))
        newT1['IA'] += -0.5*np.einsum('MNAE,MNIE->IA', self.T2['IJAB'], self.V('MNIE'))
        newT1['IA'] += -np.einsum('MnAe,MnIe->IA', self.T2['IjAb'], self.V('MnIe'))
        newT1['IA'] += 0.5*np.einsum('IMEF,AMEF->IA', self.T2['IJAB'], self.V('AMEF'))
        newT1['IA'] += np.einsum('ImEf,AmEf->IA', self.T2['IjAb'], self.V('AmEf'))
        newT1['IA'] *= self.d['IA']

        # Update T(ia)
        newT1['ia'] += self.fock_ov 
        newT1['ia'] += np.einsum('ie,ae->ia', self.T1['ia'], self.F['ae'])
        newT1['ia'] += -np.einsum('ma,mi->ia', self.T1['ia'], self.F['mi'])
        newT1['ia'] += np.einsum('imae,me->ia', self.T2['ijab'], self.F['me'])
        newT1['ia'] += np.einsum('MiEa,ME->ia', self.T2['IjAb'], self.F['ME'])
        newT1['ia'] += np.einsum('me,amie->ia', self.T1['ia'], self.V('amie'))
        newT1['ia'] += np.einsum('ME,aMiE->ia', self.T1['IA'], self.V('aMiE'))
        newT1['ia'] += -0.5*np.einsum('mnae,mnie->ia', self.T2['ijab'], self.V('mnie'))
        newT1['ia'] += -np.einsum('NmEa,mNiE->ia', self.T2['IjAb'], self.V('mNiE'))
        newT1['ia'] += 0.5*np.einsum('imef,amef->ia', self.T2['ijab'], self.V('amef'))
        newT1['ia'] += np.einsum('MiFe,aMeF->ia', self.T2['IjAb'], self.V('aMeF'))
        newT1['ia'] *= self.d['ia']

        # Update T(IJAB)

        newT2['IJAB'] += self.V('IJAB')
        X = self.T2['IJAB'] + np.einsum('MA,NB->MNAB', self.T1['IA'], self.T1['IA']) - np.einsum('MB,NA->MNAB', self.T1['IA'], self.T1['IA'])
        newT2['IJAB'] += 0.5*np.einsum('MNAB,MNIJ->IJAB', X, self.W['MNIJ'])

        X = self.T2['IJAB'] + np.einsum('IE,JF->IJEF', self.T1['IA'], self.T1['IA']) - np.einsum('IF,JE->IJEF', self.T1['IA'], self.T1['IA'])
        newT2['IJAB'] += 0.5*np.einsum('IJEF,ABEF->IJAB', X, self.W['ABEF'])

        X = self.F['AE'] - 0.5*np.einsum('MB,ME->BE', self.T1['IA'], self.F['ME'])
        Pab = np.einsum('IJAE,BE->IJAB', self.T2['IJAB'], X) - np.einsum('MA,MBIJ->IJAB', self.T1['IA'], self.V('MBIJ'))

        X = self.F['MI'] + 0.5*np.einsum('JE,ME->MJ', self.T1['IA'], self.F['ME']) 
        Pij = -np.einsum('IMAB,MJ->IJAB', self.T2['IJAB'], X) + np.einsum('IE,ABEJ->IJAB', self.T1['IA'], self.V('ABEJ'))

        Pijab  =  np.einsum('IMAE,MBEJ->IJAB', self.T2['IJAB'], self.W['MBEJ'])
        Pijab += -np.einsum('IE,MA,MBEJ->IJAB', self.T1['IA'], self.T1['IA'], self.V('MBEJ'))
        Pijab += np.einsum('ImAe,mBeJ->IJAB', self.T2['IjAb'], self.W['mBeJ'])

        newT2['IJAB'] += Pab + Pij + Pijab - (Pab + Pijab).transpose(0,1,3,2) - (Pij + Pijab).transpose(1,0,2,3) + Pijab.transpose(1,0,3,2)
         
        newT2['IJAB'] *= self.D['IJAB']

        # Update T(ijab)

        newT2['ijab'] += self.V('ijab')
        X = self.T2['ijab'] + np.einsum('ma,nb->mnab', self.T1['ia'], self.T1['ia']) - np.einsum('mb,na->mnab', self.T1['ia'], self.T1['ia'])
        newT2['ijab'] += 0.5*np.einsum('mnab,mnij->ijab', X, self.W['mnij'])

        X = self.T2['ijab'] + np.einsum('ie,jf->ijef', self.T1['ia'], self.T1['ia']) - np.einsum('if,je->ijef', self.T1['ia'], self.T1['ia'])
        newT2['ijab'] += 0.5*np.einsum('ijef,abef->ijab', X, self.W['abef'])

        X = self.F['ae'] - 0.5*np.einsum('mb,me->be', self.T1['ia'], self.F['me'])
        Pab = np.einsum('ijae,be->ijab', self.T2['ijab'], X) - np.einsum('ma,mbij->ijab', self.T1['ia'], self.V('mbij'))

        X = self.F['mi'] + 0.5*np.einsum('je,me->mj', self.T1['ia'], self.F['me']) 
        Pij = -np.einsum('imab,mj->ijab', self.T2['ijab'], X) + np.einsum('ie,abej->ijab', self.T1['ia'], self.V('abej'))

        Pijab  =  np.einsum('imae,mbej->ijab', self.T2['ijab'], self.W['mbej'])
        Pijab += -np.einsum('ie,ma,mbej->ijab', self.T1['ia'], self.T1['ia'], self.V('mbej'))
        Pijab += np.einsum('MiEa,MbEj->ijab', self.T2['IjAb'], self.W['MbEj'])

        newT2['ijab'] += Pab + Pij + Pijab - (Pab + Pijab).transpose(0,1,3,2) - (Pij + Pijab).transpose(1,0,2,3) + Pijab.transpose(1,0,3,2)

        newT2['ijab'] *= self.D['ijab']

        # Update T(IjAb)

        newT2['IjAb'] += self.V('IjAb')
        X = self.F['ae'] - 0.5*np.einsum('mb,me->be', self.T1['ia'], self.F['me'])
        newT2['IjAb'] += np.einsum('IjAe,be->IjAb', self.T2['IjAb'], X)

        X = self.F['AE'] - 0.5*np.einsum('MA,ME->AE', self.T1['IA'], self.F['ME']) 
        newT2['IjAb'] += np.einsum('IjEb,AE->IjAb', self.T2['IjAb'], X)

        X = self.F['mi'] + 0.5*np.einsum('je,me->mj', self.T1['ia'], self.F['me']) 
        newT2['IjAb'] += -np.einsum('ImAb,mj->IjAb', self.T2['IjAb'], X)

        X = self.F['MI'] + 0.5*np.einsum('IE,ME->MI', self.T1['IA'], self.F['ME']) 
        newT2['IjAb'] += -np.einsum('MjAb,MI->IjAb', self.T2['IjAb'], X)

        X = self.T2['IjAb'] + np.einsum('MA,nb->MnAb', self.T1['IA'], self.T1['ia'])
        newT2['IjAb'] += 0.5*np.einsum('MnAb,MnIj->IjAb', X, self.W['MnIj'])

        X = self.T2['IjAb'] + np.einsum('mb,NA->NmAb', self.T1['ia'], self.T1['IA'])
        newT2['IjAb'] += 0.5*np.einsum('NmAb,NmIj->IjAb', X, self.W['MnIj'])

        X = self.T2['IjAb'] + np.einsum('IE,jf->IjEf', self.T1['IA'], self.T1['ia']) 
        newT2['IjAb'] += 0.5*np.einsum('IjEf,AbEf->IjAb', X, self.W['AbEf'])

        X = self.T2['IjAb'] + np.einsum('IF,je->IjFe', self.T1['IA'], self.T1['ia']) 
        newT2['IjAb'] += 0.5*np.einsum('IjFe,AbFe->IjAb', X, self.W['AbEf'])

        newT2['IjAb'] += np.einsum('IMAE,MbEj->IjAb', self.T2['IJAB'], self.W['MbEj'])
        newT2['IjAb'] += -np.einsum('IE,MA,MbEj->IjAb', self.T1['IA'], self.T1['IA'], self.V('MbEj'))

        newT2['IjAb'] += np.einsum('ImAe,mbej->IjAb', self.T2['IjAb'], self.W['mbej'])
        
        newT2['IjAb'] += np.einsum('ImEb,mAEj->IjAb', self.T2['IjAb'], self.W['mBEj'])
        newT2['IjAb'] += np.einsum('IE,mb,mAEj->IjAb', self.T1['IA'], self.T1['ia'], self.V('mAEj'))

        newT2['IjAb'] += np.einsum('MjAe,MbeI->IjAb', self.T2['IjAb'], self.W['MbeJ'])
        newT2['IjAb'] += np.einsum('je,MA,MbeI->IjAb', self.T1['ia'], self.T1['IA'], self.V('MbeI'))

        newT2['IjAb'] += np.einsum('jmbe,mAeI->IjAb', self.T2['ijab'], self.W['mBeJ'])
        newT2['IjAb'] += -np.einsum('je,mb,mAeI->IjAb', self.T1['ia'], self.T1['ia'], self.V('mAeI'))

        newT2['IjAb'] += np.einsum('MjEb,MAEI->IjAb', self.T2['IjAb'], self.W['MBEJ'])

        newT2['IjAb'] += np.einsum('IE,AbEj->IjAb', self.T1['IA'], self.V('AbEj'))
        newT2['IjAb'] += -np.einsum('je,AbeI->IjAb', self.T1['ia'], self.V('AbeI'))
        newT2['IjAb'] += -np.einsum('MA,MbIj->IjAb', self.T1['IA'], self.V('MbIj'))
        newT2['IjAb'] += np.einsum('mb,mAIj->IjAb', self.T1['ia'], self.V('mAIj'))

        newT2['IjAb'] *= self.D['IjAb']

        # Compute RMS

        self.rms1['IA'] = np.sqrt(np.sum(np.square(newT1['IA'] - self.T1['IA'] )))/(self.nalpha*self.avir)
        self.rms1['ia'] = np.sqrt(np.sum(np.square(newT1['ia'] - self.T1['ia'] )))/(self.nbeta*self.bvir)

        self.rms2['IJAB'] = np.sqrt(np.sum(np.square(newT2['IJAB'] - self.T2['IJAB'] )))/(self.nalpha*self.nalpha*self.avir*self.avir)
        self.rms2['ijab'] = np.sqrt(np.sum(np.square(newT2['ijab'] - self.T2['ijab'] )))/(self.nbeta*self.nbeta*self.bvir*self.bvir)
        self.rms2['IjAb'] = np.sqrt(np.sum(np.square(newT2['IjAb'] - self.T2['IjAb'] )))/(self.nalpha*self.nbeta*self.avir*self.bvir)

        # Save new amplitudes

        self.T1 = newT1
        self.T2 = newT2

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

        # Save reference wavefunction properties
        self.Ehf = wfn.energy() 
        self.nmo = wfn.nmo()
        self.nalpha = wfn.nalpha() 
        self.nbeta = wfn.nbeta()
        self.avir = self.nmo - self.nalpha
        self.bvir = self.nmo - self.nbeta
        self.Ca = wfn.Ca()
        self.Cb = wfn.Cb()
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()

        # Save Options
        self.CC_CONV = CC_CONV
        self.E_CONV = E_CONV
        self.CC_MAXITER = CC_MAXITER

        print("Number of alpha electrons:   {}".format(self.nalpha))
        print("Number of beta electrons:    {}".format(self.nbeta))
        print("Number of MOs:               {}".format(self.nmo))

        print("\n Transforming integrals...")

        mints = psi4.core.MintsHelper(wfn.basisset())
        self.mints = mints
        # One electron integral
        h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        ## Alpha
        ah = np.einsum('up,vq,uv->pq', self.Ca, self.Ca, h)
        ## Beta
        bh = np.einsum('up,vq,uv->pq', self.Cb, self.Cb, h)

        Vaaaa = np.asarray(mints.mo_eri(self.Ca, self.Ca, self.Ca, self.Ca))
        Vbbbb = np.asarray(mints.mo_eri(self.Cb, self.Cb, self.Cb, self.Cb))
        Vaabb = np.asarray(mints.mo_eri(self.Ca, self.Ca, self.Cb, self.Cb))
        Vbbaa = np.asarray(mints.mo_eri(self.Cb, self.Cb, self.Ca, self.Ca))
    
        # Slices
        oa = slice(0,self.nalpha)
        va = slice(self.nalpha,self.nmo)
        ob = slice(0,self.nbeta)
        vb = slice(self.nbeta,self.nmo)

        # Form the full fock matrices
        fa = ah + np.einsum('pqkk->pq', (Vaaaa - Vaaaa.swapaxes(1,2))[:,:,oa,oa]) + np.einsum('pqkk->pq', Vaabb[:,:,ob,ob])
        fb = bh + np.einsum('pqkk->pq', (Vbbbb - Vbbbb.swapaxes(1,2))[:,:,ob,ob]) + np.einsum('pqkk->pq', Vbbaa[:,:,oa,oa])

        # Save diagonal terms
        self.fock_Od = copy.deepcopy(fa.diagonal()[oa])
        self.fock_Vd = copy.deepcopy(fa.diagonal()[va])
        self.fock_od = copy.deepcopy(fb.diagonal()[ob])
        self.fock_vd = copy.deepcopy(fb.diagonal()[vb])

        # Erase diagonal elements from original matrix
        np.fill_diagonal(fa, 0.0)
        np.fill_diagonal(fb, 0.0)

        # Save useful slices
        # Alphas
        self.fock_OO = fa[oa,oa]
        self.fock_VV = fa[va,va]
        self.fock_OV = fa[oa,va]
        # Betas
        self.fock_oo = fb[ob,ob]
        self.fock_vv = fb[vb,vb]
        self.fock_ov = fa[ob,vb]

        # Save two-electron integral in physicists notation
        self.Vaaaa = Vaaaa.swapaxes(1,2)
        self.Vaaaa = self.Vaaaa - self.Vaaaa.swapaxes(2,3)

        self.Vbbbb = Vbbbb.swapaxes(1,2)
        self.Vbbbb = self.Vbbbb - self.Vbbbb.swapaxes(2,3)

        self.Vabab = Vaabb.swapaxes(1,2)
        self.Vbaba = Vbbaa.swapaxes(1,2)

        self.Vabba = - Vaabb.transpose(0,3,2,1)
        self.Vbaab = - Vbbaa.transpose(0,3,2,1)

        self.compute()

    def compute(self):

        # Auxiliar D matrices

        new = np.newaxis
        self.d = {
        'IA' : 1.0/(self.fock_Od[:, new] - self.fock_Vd[new, :]),
        'ia' : 1.0/(self.fock_od[:, new] - self.fock_vd[new, :])
        }

        self.D = {
        'IJAB' : 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :]),
        'ijab' : 1.0/(self.fock_od[:, new, new, new] + self.fock_od[new, :, new, new] - self.fock_vd[new, new, :, new] - self.fock_vd[new, new, new, :]),
        'IjAb' : 1.0/(self.fock_Od[:, new, new, new] + self.fock_od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_vd[new, new, new, :]),
        }

        # Initial T1 amplitudes

        self.T1 = {
        'IA' : self.fock_OV*self.d['IA'],
        'ia' : self.fock_ov*self.d['ia']
        }

        # Initial T2 amplitudes

        self.T2 = {
        'IJAB' : self.D['IJAB']*self.V('IJAB'),
        'ijab' : self.D['ijab']*self.V('ijab'),
        'IjAb' : self.D['IjAb']*self.V('IjAb'),
        }

        # Get MP2 energy

        self.update_energy()

        print('MP2 Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))

        self.rms1 = {
        'IA' : 0.0,
        'ia' : 0.0
        }

        # Initial T2 amplitudes

        self.rms2 = {
        'IJAB' : 0.0,
        'ijab' : 0.0,
        'IjAb' : 0.0,
        }

        max_rms = 1
        dE = 1
        ite = 1

        rms_LIM = 10**(-self.CC_CONV)
        E_LIM = 10**(-self.E_CONV)
        f = False
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
            max_rms = max(self.rms1[max(self.rms1)], self.rms2[max(self.rms2)])
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {:< 15.10f}".format(self.Ecc))
            print("Energy change:         {:< 15.10f}".format(dE))
            print("Max RMS residue:       {:< 15.10f}".format(max_rms))
            print("Time required:         {:< 15.10f}".format(time.time() - t))
            print('='*37)
            ite += 1

        print('CC Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))
        print('CCSD iterations took %.2f seconds.\n' % (time.time() - t0))
