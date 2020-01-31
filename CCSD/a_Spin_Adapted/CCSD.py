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

        def space(entry):
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
        x,y = arg
        if x == y:

            # For repeaded argument (e.g. ii, aa), returns diagonal of Fock matrix
    
            if x.isupper():
                if x in 'IJKLMN':
                    return self.fa.diagonal()[0:self.nalpha]
                elif x in 'ABCDEF':
                    return self.fa.diagonal()[self.nalpha:self.nmo]
                else:
                    raise NameError('Invalid Fock key')
            else:
                if x in 'ijklmn':
                    return self.fb.diagonal()[0:self.nbeta]
                elif x in 'abcdef':
                    return self.fb.diagonal()[self.nbeta:self.nmo]
                else:
                    raise NameError('Invalid Fock key')
        
        elif arg.isupper():

            if x in 'IJKLMN' and y in 'IJKLMN':
                # e.g. f_MI
                out = copy.deepcopy(self.fa[0:self.nalpha, 0:self.nalpha])
                np.fill_diagonal(out, 0.0)
                return out

            elif x in 'IJKLMN' and y in 'ABCDEF':
                # e.g. f_ME
                out = copy.deepcopy(self.fa[0:self.nalpha, self.nalpha:self.nmo])
                np.fill_diagonal(out, 0.0)
                return out

            elif x in 'ABCDEF' and y in 'ABCDEF':
                # e.g. f_AE
                out = copy.deepcopy(self.fa[self.nalpha:self.nmo, self.nalpha:self.nmo])
                np.fill_diagonal(out, 0.0)
                return out
            else:
                raise NameError('Invalid Fock key')

        elif arg.islower():

            if x in 'ijklmn' and y in 'ijklmn':
                # e.g. f_mi
                out = copy.deepcopy(self.fb[0:self.nbeta, 0:self.nbeta])
                np.fill_diagonal(out, 0.0)
                return out

            elif x in 'ijklmn' and y in 'abcdef':
                # e.g. f_me
                out = copy.deepcopy(self.fb[0:self.nbeta, self.nbeta:self.nmo])
                np.fill_diagonal(out, 0.0)
                return out

            elif x in 'abcdef' and y in 'abcdef':
                # e.g. f_ae
                out = copy.deepcopy(self.fb[self.nbeta:self.nmo, self.nbeta:self.nmo])
                np.fill_diagonal(out, 0.0)
                return out
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
        self.F.update({'AE' : np.zeros((self.nmo - self.nalpha, self.nmo - self.nalpha))})
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
        self.F.update({'ae' : np.zeros((self.nmo - self.nbeta, self.nmo - self.nbeta))})
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
        

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):
        self.Ehf = wfn.energy() 
        self.nalpha = wfn.nalpha() 
        self.nbeta = wfn.nbeta()
        self.Ca = wfn.Ca()
        self.Cb = wfn.Cb()
        self.fda = np.asarray(wfn.epsilon_a())
        self.fdb = np.asarray(wfn.epsilon_b())
        self.nmo = wfn.nmo()
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

        # Save matrices in physicists notation
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

        self.update_energy()
        self.update_Fint()
        print(self.Ecc)


