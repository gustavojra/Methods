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

    def f(self, arg):

        # Auxiliar method to handle different Fock matricex (i.e. virtual, occipied, alpha, beta)
        # This function is called as self.f(string)
        # where string contains two characters representing the two indices as
        # -> Capital letters for alpha orbitals
        # -> Lower case letters for beta orbitals 
        # -> ijklmn for occupied orbitals
        # -> abcdef for virtual orbitals
        # -> Repeated arguments (e.g. self.f('II')) returns diagoal elements of Fock matrix within that orbital/spin case
        # -> Different arguments (e.g. self.f('ia')) returns the off-diagonal elements of the Fock matrix. 
        # No mixed spin is allowed (e.g. self.f('Ia'))

        x,y = arg
        if x == y:

            # For repeated argument (e.g. ii, aa), returns diagonal of Fock matrix
    
            if x.isupper():
                if x in 'IJKLMN':
                    return self.fa_d[0:self.nalpha]
                elif x in 'ABCDEF':
                    return self.fa_d[self.nalpha:self.nmo]
                else:
                    raise NameError('Invalid Fock key')
            else:
                if x in 'ijklmn':
                    return self.fb_d[0:self.nbeta]
                elif x in 'abcdef':
                    return self.fb_d[self.nbeta:self.nmo]
                else:
                    raise NameError('Invalid Fock key')
        
        elif arg.isupper():

            if x in 'IJKLMN' and y in 'IJKLMN':
                # e.g. f_MI
                return self.fa[0:self.nalpha, 0:self.nalpha]

            elif x in 'IJKLMN' and y in 'ABCDEF':
                # e.g. f_ME
                return self.fa[0:self.nalpha, self.nalpha:self.nmo]

            elif x in 'ABCDEF' and y in 'ABCDEF':
                # e.g. f_AE
                return self.fa[self.nalpha:self.nmo, self.nalpha:self.nmo]

            else:
                raise NameError('Invalid Fock key')

        elif arg.islower():

            if x in 'ijklmn' and y in 'ijklmn':
                # e.g. f_mi
                return self.fb[0:self.nbeta, 0:self.nbeta]

            elif x in 'ijklmn' and y in 'abcdef':
                # e.g. f_me
                return self.fb[0:self.nbeta, self.nbeta:self.nmo]

            elif x in 'abcdef' and y in 'abcdef':
                # e.g. f_ae
                return self.fb[self.nbeta:self.nmo, self.nbeta:self.nmo]

            else:
                raise NameError('Invalid Fock key')
        else:
            raise NameError('Invalid Fock key')
            
    def T(self, arg):
        if len(arg) == 2:

            # Calling T1 amplitudes
            if arg.isupper():
                return self.T1amp['IA']
            elif arg.islower():
                return self.T1amp['ia']
            else:
                raise NameError('Invalid T1 key')

        if len(arg) == 4:

            # Calling T2 amplitudes
            if arg.isupper():
                return self.T2amp['IJAB']
            elif arg.islower():
                return self.T2amp['ijab']
            elif (arg[0] + arg[2]).isupper():
                return self.T2amp['IjAb']
            elif (arg[0] + arg[2]).islower():
                return self.T2amp['iJaB']
            elif (arg[0] + arg[3]).isupper():
                return self.T2amp['IjaB'] 
            elif (arg[0] + arg[3]).islower():
                return self.T2amp['iJAb']
            else:
                raise NameError('Invalid T2 key')

        else:
            raise NameError('Invalid amplitude key')

    def update_energy(self):
        
        self.Ecc  = np.einsum('IA,IA->', self.f('IA'), self.T('IA'))
        self.Ecc += np.einsum('ia,ia->', self.f('ia'), self.T('ia'))
    
        X = self.T('IJAB') + 2*np.einsum('IA,JB->IJAB', self.T('IA'), self.T('JB'))
        self.Ecc += (1.0/4.0)*np.einsum('IJAB,IJAB->', X, self.V('IJAB'))
    
        X = self.T('ijab') + 2*np.einsum('ia,jb->ijab', self.T('ia'), self.T('jb'))
        self.Ecc += (1.0/4.0)*np.einsum('ijab,ijab->', X, self.V('ijab'))
    
        X = self.T('IjAb') + 2*np.einsum('IA,jb->IjAb', self.T('IA'), self.T('jb'))
        self.Ecc += (1.0/4.0)*np.einsum('IjAb,IjAb->', X, self.V('IjAb'))
    
        X = self.T('iJaB') + 2*np.einsum('ia,JB->iJaB', self.T('ia'), self.T('JB'))
        self.Ecc += (1.0/4.0)*np.einsum('iJaB,iJaB->', X, self.V('iJaB'))
    
        self.Ecc += (1.0/4.0)*np.einsum('IjaB,IjaB->', self.T('IjaB'), self.V('IjaB'))
    
        self.Ecc += (1.0/4.0)*np.einsum('iJAb,iJAb->', self.T('iJAb'), self.V('iJAb'))

    def update_Fint(self):

        # Clean up F cases
        self.F = {}

        # Update F(AE)
        self.F.update({'AE' : np.zeros((self.avir, self.avir))})
        self.F['AE'] += self.f('AE') - 0.5*np.einsum('ME,MA->AE', self.f('ME'), self.T('MA'))
        self.F['AE'] += np.einsum('MF,AMEF->AE', self.T('MF'), self.V('AMEF'))
        self.F['AE'] += np.einsum('mf,AmEf->AE', self.T('mf'), self.V('AmEf'))

        X = self.T('MNAF') + 0.5*np.einsum('MA,NF->MNAF', self.T('MA'), self.T('NF')) - 0.5*np.einsum('MF,NA->MNAF', self.T('MF'), self.T('NA'))
        self.F['AE'] += -0.5*np.einsum('MNAF,MNEF->AE', X, self.V('MNEF'))

        X = self.T('MnAf') + 0.5*np.einsum('MA,nf->MnAf', self.T('MA'), self.T('nf'))
        self.F['AE'] += -0.5*np.einsum('MnAf,MnEf->AE', X, self.V('MnEf'))

        X = self.T('mNAf') - 0.5*np.einsum('mf,NA->mNAf', self.T('mf'), self.T('NA'))
        self.F['AE'] += -0.5*np.einsum('mNAf,mNEf->AE', X, self.V('mNEf'))

        # Update F(ae)
        self.F.update({'ae' : np.zeros((self.bvir, self.bvir))})
        self.F['ae'] += self.f('ae') - 0.5*np.einsum('me,ma->ae', self.f('me'), self.T('ma'))
        self.F['ae'] += np.einsum('mf,amef->ae', self.T('mf'), self.V('amef'))
        self.F['ae'] += np.einsum('MF,aMeF->ae', self.T('MF'), self.V('aMeF'))

        X = self.T('mnaf') + 0.5*np.einsum('ma,nf->mnaf', self.T('ma'), self.T('nf')) - 0.5*np.einsum('mf,na->mnaf', self.T('mf'), self.T('na'))
        self.F['ae'] += -0.5*np.einsum('mnaf,mnef->ae', X, self.V('mnef'))

        X = self.T('mNaF') + 0.5*np.einsum('ma,NF->mNaF', self.T('ma'), self.T('NF'))
        self.F['ae'] += -0.5*np.einsum('mNaF,mNeF->ae', X, self.V('mNeF'))

        X = self.T('MnaF') - 0.5*np.einsum('MF,na->MnaF', self.T('MF'), self.T('na'))
        self.F['ae'] += -0.5*np.einsum('MnaF,MneF->ae', X, self.V('MneF'))

        # Update F(MI)
        self.F.update({'MI' : np.zeros((self.nalpha, self.nalpha))})
        self.F['MI'] += self.f('MI') + 0.5*np.einsum('ME,IE->MI', self.f('ME'), self.T('IE'))
        self.F['MI'] += np.einsum('NE,MNIE->MI', self.T('NE'), self.V('MNIE'))
        self.F['MI'] += np.einsum('ne,MnIe->MI', self.T('ne'), self.V('MnIe'))

        X = self.T('INEF') + 0.5*np.einsum('IE,NF->INEF', self.T('IE'), self.T('NF')) - 0.5*np.einsum('IF,NE->INEF', self.T('IF'), self.T('NE'))
        self.F['MI'] += +0.5*np.einsum('INEF,MNEF->MI', X, self.V('MNEF'))

        X = self.T('InEf') + 0.5*np.einsum('IE,nf->InEf', self.T('IE'), self.T('nf'))
        self.F['MI'] += +0.5*np.einsum('InEf,MnEf', X, self.V('MnEf'))

        X = self.T('IneF') - 0.5*np.einsum('IF,ne->IneF', self.T('IF'), self.T('ne'))
        self.F['MI'] += +0.5*np.einsum('IneF,MneF->MI', X, self.V('MneF'))

        # Update F(mi)
        self.F.update({'mi' : np.zeros((self.nbeta, self.nbeta))})
        self.F['mi'] += self.f('mi') + 0.5*np.einsum('me,ie->mi', self.f('me'), self.T('ie'))
        self.F['mi'] += np.einsum('ne,mnie->mi', self.T('ne'), self.V('mnie'))
        self.F['mi'] += np.einsum('NE,mNiE->mi', self.T('NE'), self.V('mNiE'))

        X = self.T('inef') + 0.5*np.einsum('ie,nf->inef', self.T('ie'), self.T('nf')) - 0.5*np.einsum('if,ne->inef', self.T('if'), self.T('ne'))
        self.F['mi'] += +0.5*np.einsum('inef,mnef->mi', X, self.V('mnef'))

        X = self.T('iNeF') + 0.5*np.einsum('ie,NF->iNeF', self.T('ie'), self.T('NF'))
        self.F['mi'] += +0.5*np.einsum('iNeF,mNeF->mi', X, self.V('mNeF'))

        X = self.T('iNEf') - 0.5*np.einsum('if,NE->iNEf', self.T('if'), self.T('NE'))
        self.F['mi'] += +0.5*np.einsum('iNEf,mNEf->mi', X, self.V('mNEf'))

        # Update F(ME)
        self.F.update({'ME' : np.zeros((self.nalpha, self.avir))})
        self.F['ME'] += self.f('ME') + np.einsum('NF, MNEF-> ME', self.T('NF'), self.V('MNEF')) + np.einsum('nf, MnEf-> ME', self.T('nf'), self.V('MnEf'))

        # Update F(me)
        self.F.update({'me' : np.zeros((self.nbeta, self.bvir))})
        self.F['me'] += self.f('me') + np.einsum('nf, mnef-> me', self.T('nf'), self.V('mnef')) + np.einsum('NF, mNeF-> me', self.T('NF'), self.V('mNeF'))

    def update_Winf(self):

        # Clean up W cases
        self.W = {}

        # Update W(MNIJ)
        self.W.update({'MNIJ' : np.zeros((self.nalpha, self.nalpha, self.nalpha, self.nalpha))})
        self.W['MNIJ'] += self.V('MNIJ')
        self.W['MNIJ'] += np.einsum('JE, MNIE-> MNIJ', self.T('JE'), self.V('MNIE'))
        self.W['MNIJ'] += -np.einsum('IE, MNJE-> MNIJ', self.T('IE'), self.V('MNJE'))
        X = self.T('IJEF') + np.einsum('IE,JF->IJEF', self.T('IE'), self.T('JF')) - np.einsum('IF,JE->IJEF', self.T('IF'), self.T('JF'))
        self.W['MNIJ'] += (1.0/4.0)*np.einsum('IJEF,MNEF->MNIJ', X, self.V('MNEF'))

        # Update W(MnIj)
        self.W.update({'MnIj' : np.zeros((self.nalpha, self.nbeta, self.nalpha, self.nbeta))})
        self.W['MnIj'] += self.V('MnIj')
        self.W['MnIj'] += np.einsum('je, MnIe-> MnIj', self.T('je'), self.V('MnIe'))
        self.W['MnIj'] += -np.einsum('IE,MnjE -> MnIj', self.T('IE'), self.V('MnjE'))
        X = self.T('IjEf') + np.einsum('IE,jf->IjEf', self.T('IE'), self.T('jf')) 
        self.W['MnIj'] += (1.0/4.0)*np.einsum('IjEf,MnEf->MnIj', X, self.V('MnEf'))
        X = self.T('IjeF') - np.einsum('IF,je->IjeF', self.T('IF'), self.T('je')) 
        self.W['MnIj'] += (1.0/4.0)*np.einsum('IjeF,MneF->MnIj', X, self.V('MneF'))

        # Update W(MniJ)
        self.W.update({'MniJ' : np.zeros((self.nalpha, self.nbeta, self.nbeta, self.nalpha))})
        self.W['MniJ'] += self.V('MniJ')
        self.W['MniJ'] += np.einsum('JE, MniE-> MniJ', self.T('JE'), self.V('MniE'))
        self.W['MniJ'] += -np.einsum('ie,MnJe -> MniJ', self.T('ie'), self.V('MnJe'))
        X = self.T('iJeF') + np.einsum('ie,JF->iJeF', self.T('ie'), self.T('JF')) 
        self.W['MniJ'] += (1.0/4.0)*np.einsum('iJeF,MneF->MniJ', X, self.V('MneF'))
        X = self.T('iJEf') - np.einsum('if,JE->iJEf', self.T('if'), self.T('JE')) 
        self.W['MniJ'] += (1.0/4.0)*np.einsum('iJEf,MnEf->MniJ', X, self.V('MnEf'))

        # Update W(mnij)
        self.W.update({'mnij' : np.zeros((self.nbeta, self.nbeta, self.nbeta, self.nbeta))})
        self.W['mnij'] += self.V('mnij')
        self.W['mnij'] += np.einsum('je, mnie-> mnij', self.T('je'), self.V('mnie'))
        self.W['mnij'] += -np.einsum('ie, mnje-> mnij', self.T('ie'), self.V('mnje'))
        X = self.T('ijef') + np.einsum('ie,jf->ijef', self.T('ie'), self.T('jf')) - np.einsum('if,je->ijef', self.T('if'), self.T('jf'))
        self.W['mnij'] += (1.0/4.0)*np.einsum('ijef,mnef->mnij', X, self.V('mnef'))

        # Update W(mNiJ)
        self.W.update({'mNiJ' : np.zeros((self.nbeta, self.nalpha, self.nbeta, self.nalpha))})
        self.W['mNiJ'] += self.V('mNiJ')
        self.W['mNiJ'] += np.einsum('JE, mNiE-> mNiJ', self.T('JE'), self.V('mNiE'))
        self.W['mNiJ'] += -np.einsum('ie,mNJe -> mNiJ', self.T('ie'), self.V('mNJe'))
        X = self.T('iJeF') + np.einsum('ie,JF->iJeF', self.T('ie'), self.T('JF')) 
        self.W['mNiJ'] += (1.0/4.0)*np.einsum('iJeF,mNeF->mNiJ', X, self.V('mNeF'))
        X = self.T('iJEf') - np.einsum('if,JE->iJEf', self.T('if'), self.T('JE')) 
        self.W['mNiJ'] += (1.0/4.0)*np.einsum('iJEf,mNEf->mNiJ', X, self.V('mNEf'))

        # Update W(mNIj)
        self.W.update({'mNIj' : np.zeros((self.nbeta, self.nalpha, self.nalpha, self.nbeta))})
        self.W['mNIj'] += self.V('mNIj')
        self.W['mNIj'] += np.einsum('je, mNIe-> mNIj', self.T('je'), self.V('mNIe'))
        self.W['mNIj'] += -np.einsum('IE,mNjE -> mNIj', self.T('IE'), self.V('mNjE'))
        X = self.T('IjEf') + np.einsum('IE,jf->IjEf', self.T('IE'), self.T('jf')) 
        self.W['mNIj'] += (1.0/4.0)*np.einsum('IjEf,mNEf->mNIj', X, self.V('mNEf'))
        X = self.T('IjeF') - np.einsum('IF,je->IjeF', self.T('IF'), self.T('je')) 
        self.W['mNIj'] += (1.0/4.0)*np.einsum('IjeF,mNeF->mNIj', X, self.V('mNeF'))

        # Update W(ABEF)
        self.W.update({'ABEF' : np.zeros((self.avir, self.avir, self.avir, self.avir))})
        self.W['ABEF'] += self.V('ABEF')
        self.W['ABEF'] += -np.einsum('MB, AMEF-> ABEF', self.T('MB'), self.V('AMEF'))
        self.W['ABEF'] += np.einsum('MA, BMEF-> ABEF', self.T('MA'), self.V('BMEF'))
        X = self.T('MNAB') + np.einsum('MA,NB->MNAB', self.T('MA'), self.T('NB')) - np.einsum('MB,NA->MNAB', self.T('MB'), self.T('NA'))
        self.W['ABEF'] += (1.0/4.0)*np.einsum('MNAB,MNEF->ABEF', X, self.V('MNEF'))

        # Update W(abef)
        self.W.update({'abef' : np.zeros((self.bvir, self.bvir, self.bvir, self.bvir))})
        self.W['abef'] += self.V('abef')
        self.W['abef'] += -np.einsum('mb, amef-> abef', self.T('mb'), self.V('amef'))
        self.W['abef'] += np.einsum('ma, bmef-> abef', self.T('ma'), self.V('bmef'))
        X = self.T('mnab') + np.einsum('ma,nb->mnab', self.T('ma'), self.T('nb')) - np.einsum('mb,na->mnab', self.T('mb'), self.T('na'))
        self.W['abef'] += (1.0/4.0)*np.einsum('mnab,mnef->abef', X, self.V('mnef'))

        # Update W(AbEf)
        self.W.update({'AbEf' : np.zeros((self.avir, self.bvir, self.avir, self.bvir))})
        self.W['AbEf'] += self.V('AbEf')
        self.W['AbEf'] += -np.einsum('mb, AmEf-> AbEf', self.T('mb'), self.V('AmEf'))
        self.W['AbEf'] += np.einsum('MA,bMEf -> AbEf', self.T('MA'), self.V('bMEf'))
        X = self.T('MnAb') + np.einsum('MA,nb->MnAb', self.T('MA'), self.T('nb')) 
        self.W['AbEf'] += (1.0/4.0)*np.einsum('MnAb,MnEf->AbEf', X, self.V('MnEf'))
        X = self.T('mNAb') - np.einsum('mb,NA->mNAb', self.T('mb'), self.T('NA')) 
        self.W['AbEf'] += (1.0/4.0)*np.einsum('mNAb,mNEf->AbEf', X, self.V('mNEf'))

        # Update W(AbeF)
        self.W.update({'AbeF' : np.zeros((self.avir, self.bvir, self.bvir, self.avir))})
        self.W['AbeF'] += self.V('AbeF')
        self.W['AbeF'] += -np.einsum('mb, AmeF-> AbeF', self.T('mb'), self.V('AmeF'))
        self.W['AbeF'] += np.einsum('MA,bMeF -> AbeF', self.T('MA'), self.V('bMeF'))
        X = self.T('MnAb') + np.einsum('MA,nb->MnAb', self.T('MA'), self.T('nb')) 
        self.W['AbeF'] += (1.0/4.0)*np.einsum('MnAb,MneF->AbeF', X, self.V('MneF'))
        X = self.T('mNAb') - np.einsum('mb,NA->mNAb', self.T('mb'), self.T('NA')) 
        self.W['AbeF'] += (1.0/4.0)*np.einsum('mNAb,mNeF->AbeF', X, self.V('mNeF'))

        # Update W(aBeF)
        self.W.update({'aBeF' : np.zeros((self.bvir, self.avir, self.bvir, self.avir))})
        self.W['aBeF'] += self.V('aBeF')
        self.W['aBeF'] += -np.einsum('MB, aMeF-> aBeF', self.T('MB'), self.V('aMeF'))
        self.W['aBeF'] += np.einsum('ma,BmeF -> aBeF', self.T('ma'), self.V('BmeF'))
        X = self.T('mNaB') + np.einsum('ma,NB->mNaB', self.T('ma'), self.T('NB')) 
        self.W['aBeF'] += (1.0/4.0)*np.einsum('mNaB,mNeF->aBeF', X, self.V('mNeF'))
        X = self.T('MnaB') - np.einsum('MB,na->MnaB', self.T('MB'), self.T('na')) 
        self.W['aBeF'] += (1.0/4.0)*np.einsum('MnaB,MneF->aBeF', X, self.V('MneF'))

        # Update W(aBEf)
        self.W.update({'aBEf' : np.zeros((self.bvir, self.avir, self.avir, self.bvir))})
        self.W['aBEf'] += self.V('aBEf')
        self.W['aBEf'] += -np.einsum('MB, aMEf-> aBEf', self.T('MB'), self.V('aMEf'))
        self.W['aBEf'] += np.einsum('ma,BmEf -> aBEf', self.T('ma'), self.V('BmEf'))
        X = self.T('mNaB') + np.einsum('ma,NB->mNaB', self.T('ma'), self.T('NB')) 
        self.W['aBEf'] += (1.0/4.0)*np.einsum('mNaB,mNEf->aBEf', X, self.V('mNEf'))
        X = self.T('MnaB') - np.einsum('MB,na->MnaB', self.T('MB'), self.T('na')) 
        self.W['aBEf'] += (1.0/4.0)*np.einsum('MnaB,MnEf->aBEf', X, self.V('MnEf'))

        # Update W(MBEJ)
        self.W.update({'MBEJ' : np.zeros((self.nalpha, self.avir, self.avir, self.nalpha))})
        self.W['MBEJ'] += self.V('MBEJ')
        self.W['MBEJ'] += np.einsum('JF,MBEF->MBEJ', self.T('JF'), self.V('MBEF'))
        self.W['MBEJ'] += -np.einsum('NB,MNEJ->MBEJ', self.T('NB'), self.V('MNEJ'))
        X = 0.5*self.T('JNFB') + np.einsum('JF,NB->JNFB', self.T('JF'), self.T('NB'))
        self.W['MBEJ'] += -np.einsum('JNFB,MNEF->MBEJ', X, self.V('MNEF'))
        self.W['MBEJ'] += -0.5*np.einsum('JnfB,MnEf->MBEJ', self.T('JnfB'), self.V('MnEf'))

        # Update W(mbej)
        self.W.update({'mbej' : np.zeros((self.nbeta, self.bvir, self.bvir, self.nbeta))})
        self.W['mbej'] += self.V('mbej')
        self.W['mbej'] += np.einsum('jf,mbef->mbej', self.T('jf'), self.V('mbef'))
        self.W['mbej'] += -np.einsum('nb,mnej->mbej', self.T('nb'), self.V('mnej'))
        X = 0.5*self.T('jnfb') + np.einsum('jf,nb->jnfb', self.T('jf'), self.T('nb'))
        self.W['mbej'] += -np.einsum('jnfb,mnef->mbej', X, self.V('mnef'))
        self.W['mbej'] += -0.5*np.einsum('jNFb,mNeF->mbej', self.T('jNFb'), self.V('mNeF'))

        # Update W(MbEj)
        self.W.update({'MbEj' : np.zeros((self.nalpha, self.bvir, self.avir, self.nbeta))})
        self.W['MbEj'] += self.V('MbEj')
        self.W['MbEj'] += np.einsum('jf,MbEf->MbEj', self.T('jf'), self.V('MbEf'))
        self.W['MbEj'] += -np.einsum('nb,MnEj->MbEj', self.T('nb'), self.V('MnEj'))
        X = 0.5*self.T('jnfb') + np.einsum('jf,nb->jnfb', self.T('jf'), self.T('nb'))
        self.W['MbEj'] += -np.einsum('jnfb,MnEf->MbEj', X, self.V('MnEf'))
        self.W['MbEj'] += -0.5*np.einsum('jNFb,MNEF->MbEj', self.T('jNFb'), self.V('MNEF'))

        # Update W(MbeJ)
        self.W.update({'MbeJ' : np.zeros((self.nalpha, self.bvir, self.bvir, self.nalpha))})
        self.W['MbeJ'] += self.V('MbeJ')
        self.W['MbeJ'] += np.einsum('JF,MbeF->MbeJ', self.T('JF'), self.V('MbeF'))
        self.W['MbeJ'] += -np.einsum('nb,MneJ->MbeJ', self.T('nb'), self.V('MneJ'))
        X = 0.5*self.T('JnFb') + np.einsum('JF,nb->JnFb', self.T('JF'), self.T('nb'))
        self.W['MbeJ'] += -np.einsum('JnFb,MneF->MbeJ', X, self.V('MneF'))

        # Update W(mBeJ)
        self.W.update({'mBeJ' : np.zeros((self.nbeta, self.avir, self.bvir, self.nalpha))})
        self.W['mBeJ'] += self.V('mBeJ')
        self.W['mBeJ'] += np.einsum('JF,mBeF->mBeJ', self.T('JF'), self.V('mBeF'))
        self.W['mBeJ'] += -np.einsum('NB,mNeJ->mBeJ', self.T('NB'), self.V('mNeJ'))
        X = 0.5*self.T('JNFB') + np.einsum('JF,NB->JNFB', self.T('JF'), self.T('NB'))
        self.W['mBeJ'] += -np.einsum('JNFB,mNeF->mBeJ', X, self.V('mNeF'))
        self.W['mBeJ'] += -0.5*np.einsum('JnfB,mnef->mBeJ', self.T('JnfB'), self.V('mnef'))

        # Update W(mBEj)
        self.W.update({'mBEj' : np.zeros((self.nbeta, self.avir, self.avir, self.nbeta))})
        self.W['mBEj'] += self.V('mBEj')
        self.W['mBEj'] += np.einsum('jf,mBEf->mBEj', self.T('jf'), self.V('mBEf'))
        self.W['mBEj'] += -np.einsum('NB,mNEj->mBEj', self.T('NB'), self.V('mNEj'))
        X = 0.5*self.T('jNfB') + np.einsum('jf,NB->jNfB', self.T('jf'), self.T('NB'))
        self.W['mBEj'] += -np.einsum('jNfB,mNEf->mBEj', X, self.V('mNEf'))

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):
        self.Ehf = wfn.energy() 
        self.nmo = wfn.nmo()
        self.nalpha = wfn.nalpha() 
        self.nbeta = wfn.nbeta()
        self.avir = self.nmo - self.nalpha
        self.bvir = self.nmo - self.nbeta
        self.Ca = wfn.Ca()
        self.Cb = wfn.Cb()
        self.fda = np.asarray(wfn.epsilon_a())
        self.fdb = np.asarray(wfn.epsilon_b())
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()

        print("Number of alpha electrons:   {}".format(self.nalpha))
        print("Number of beta electrons:    {}".format(self.nbeta))
        print("Number of MOs:               {}".format(self.nmo))

        print("\n Transforming integrals...")

        self.ERI = {}
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

        # Form fock matrices
        self.fa = ah + np.einsum('pqkk->pq', (Vaaaa - Vaaaa.swapaxes(1,2))[:,:,oa,oa]) + np.einsum('pqkk->pq', Vaabb[:,:,ob,ob])
        self.fb = bh + np.einsum('pqkk->pq', (Vbbbb - Vbbbb.swapaxes(1,2))[:,:,ob,ob]) + np.einsum('pqkk->pq', Vbbaa[:,:,oa,oa])
        self.fa_d = copy.deepcopy(self.fa.diagonal())
        self.fb_d = copy.deepcopy(self.fb.diagonal())
        np.fill_diagonal(self.fa, 0.0)
        np.fill_diagonal(self.fb, 0.0)

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
        'IA' : 1.0/(self.f('II')[:, new] - self.f('AA')[new, :]),
        'ia' : 1.0/(self.f('ii')[:, new] - self.f('aa')[new, :])
        }

        self.D = {
        'IJAB' : 1.0/(self.f('II')[:, new, new, new] + self.f('JJ')[new, :, new, new] - self.f('AA')[new, new, :, new] - self.f('BB')[new, new, new, :]),
        'ijab' : 1.0/(self.f('ii')[:, new, new, new] + self.f('jj')[new, :, new, new] - self.f('aa')[new, new, :, new] - self.f('bb')[new, new, new, :]),
        'IjAb' : 1.0/(self.f('II')[:, new, new, new] + self.f('jj')[new, :, new, new] - self.f('AA')[new, new, :, new] - self.f('bb')[new, new, new, :]),
        'IjaB' : 1.0/(self.f('II')[:, new, new, new] + self.f('jj')[new, :, new, new] - self.f('aa')[new, new, :, new] - self.f('BB')[new, new, new, :]),
        'iJaB' : 1.0/(self.f('ii')[:, new, new, new] + self.f('JJ')[new, :, new, new] - self.f('aa')[new, new, :, new] - self.f('BB')[new, new, new, :]),
        'iJAb' : 1.0/(self.f('ii')[:, new, new, new] + self.f('JJ')[new, :, new, new] - self.f('AA')[new, new, :, new] - self.f('bb')[new, new, new, :])
        }

        # Initial T1 amplitudes

        self.T1amp = {
        'IA' : self.f('IA')*self.d['IA'],
        'ia' : self.f('ia')*self.d['ia']
        }

        # Initial T2 amplitudes

        self.T2amp = {
        'IJAB' : self.D['IJAB']*self.V('IJAB'),
        'ijab' : self.D['ijab']*self.V('ijab'),
        'IjAb' : self.D['IjAb']*self.V('IjAb'),
        'IjaB' : self.D['IjaB']*self.V('IjaB'),
        'iJaB' : self.D['iJaB']*self.V('iJaB'),
        'iJAb' : self.D['iJAb']*self.V('iJAb')
        }

        # Get MP2 energy

        Emp2 = self.Ehf
        Emp2 += (1.0/4.0)*np.einsum('ijab,ijab->', self.T('IJAB'), self.V('IJAB'))
        Emp2 += (1.0/4.0)*np.einsum('ijab,ijab->', self.T('ijab'), self.V('ijab'))
        Emp2 += np.einsum('ijab,ijab->', self.T('IjAb'), self.V('IjAb'))

        print('MP2 Energy:   {:<15.10f}'.format(Emp2))

        for i in range(0,21):
            self.update_Fint()
            self.update_Winf()        
            self.update_energy()
            print(self.Ecc + self.Ehf)
