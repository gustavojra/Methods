import psi4
import os
import sys
import numpy as np
import time
import copy

sys.path.append('../../Aux')
from tools import *

np.set_printoptions(suppress=True, linewidth=120)

class UCCD:

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
                return -self.Vabab.transpose(0,1,3,2)[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

        # Case ba--
        elif y.isupper():
            # Case baab
            if z.isupper():
                return -self.Vabab.transpose(1,0,2,3)[space(x), space(y), space(z), space(w)]
            # Case baba
            if w.isupper():
                return self.Vabab.transpose(1,0,3,2)[space(x), space(y), space(z), space(w)]
            else:
                raise NameError('Invalid integral key')

    def update_energy(self):
        
        self.Ecc  = (1.0/4.0)*np.einsum('IJAB,IJAB->', self.T2['IJAB'], self.V('IJAB'))
        self.Ecc += (1.0/4.0)*np.einsum('ijab,ijab->', self.T2['ijab'], self.V('ijab'))
        self.Ecc += np.einsum('IjAb,IjAb->', self.T2['IjAb'], self.V('IjAb'))
    
    
    def update_Fint(self):

        # Clean up F cases
        self.F = {}

        # Update F(AE)
        self.F.update({'AE' : np.zeros((self.avir, self.avir))})
        self.F['AE'] += self.fock_VV 
        self.F['AE'] += -0.5*np.einsum('MNAF,MNEF->AE', self.T2['IJAB'], self.V('MNEF'))
        self.F['AE'] += -np.einsum('MnAf,MnEf->AE', self.T2['IjAb'], self.V('MnEf'))

        # Update F(ae)
        self.F.update({'ae' : np.zeros((self.bvir, self.bvir))})
        self.F['ae'] += self.fock_vv 
        self.F['ae'] += -0.5*np.einsum('mnaf,mnef->ae', self.T2['ijab'], self.V('mnef'))
        self.F['ae'] += -np.einsum('NmFa,mNeF->ae', self.T2['IjAb'], self.V('mNeF'))

        # Update F(MI)
        self.F.update({'MI' : np.zeros((self.nalpha, self.nalpha))})
        self.F['MI'] += self.fock_OO 
        self.F['MI'] += +0.5*np.einsum('INEF,MNEF->MI', self.T2['IJAB'], self.V('MNEF'))
        self.F['MI'] += np.einsum('InEf,MnEf->MI', self.T2['IjAb'], self.V('MnEf'))

        # Update F(mi)
        self.F.update({'mi' : np.zeros((self.nbeta, self.nbeta))})
        self.F['mi'] += self.fock_oo 
        self.F['mi'] += +0.5*np.einsum('inef,mnef->mi', self.T2['ijab'], self.V('mnef'))
        self.F['mi'] += np.einsum('NiFe,mNeF->mi', self.T2['IjAb'], self.V('mNeF'))

        # Update F(ME)
        self.F.update({'ME' : np.zeros((self.nalpha, self.avir))})
        self.F['ME'] += self.fock_OV 

        # Update F(me)
        self.F.update({'me' : np.zeros((self.nbeta, self.bvir))})
        self.F['me'] += self.fock_ov 

    def update_Winf(self):

        # Clean up W cases
        self.W = {}

        # Update W(MNIJ)
        self.W.update({'MNIJ' : np.zeros((self.nalpha, self.nalpha, self.nalpha, self.nalpha))})
        self.W['MNIJ'] += self.V('MNIJ')
        self.W['MNIJ'] += (1.0/4.0)*np.einsum('IJEF,MNEF->MNIJ', self.T2['IJAB'], self.V('MNEF'))

        # Update W(mnij)
        self.W.update({'mnij' : np.zeros((self.nbeta, self.nbeta, self.nbeta, self.nbeta))})
        self.W['mnij'] += self.V('mnij')
        self.W['mnij'] += (1.0/4.0)*np.einsum('ijef,mnef->mnij', self.T2['ijab'], self.V('mnef'))

        # Update W(MnIj)
        self.W.update({'MnIj' : np.zeros((self.nalpha, self.nbeta, self.nalpha, self.nbeta))})
        self.W['MnIj'] += self.V('MnIj')
        self.W['MnIj'] += (1.0/2.0)*np.einsum('IjEf,MnEf->MnIj', self.T2['IjAb'], self.V('MnEf'))

        # Update W(ABEF)
        self.W.update({'ABEF' : np.zeros((self.avir, self.avir, self.avir, self.avir))})
        self.W['ABEF'] += self.V('ABEF')
        self.W['ABEF'] += (1.0/4.0)*np.einsum('MNAB,MNEF->ABEF', self.T2['IJAB'], self.V('MNEF'))

        # Update W(abef)
        self.W.update({'abef' : np.zeros((self.bvir, self.bvir, self.bvir, self.bvir))})
        self.W['abef'] += self.V('abef')
        self.W['abef'] += (1.0/4.0)*np.einsum('mnab,mnef->abef', self.T2['ijab'], self.V('mnef'))

        # Update W(AbEf)
        self.W.update({'AbEf' : np.zeros((self.avir, self.bvir, self.avir, self.bvir))})
        self.W['AbEf'] += self.V('AbEf')
        self.W['AbEf'] += (1.0/2.0)*np.einsum('MnAb,MnEf->AbEf', self.T2['IjAb'], self.V('MnEf'))

        # Update W(MBEJ)
        self.W.update({'MBEJ' : np.zeros((self.nalpha, self.avir, self.avir, self.nalpha))})
        self.W['MBEJ'] += self.V('MBEJ')
        self.W['MBEJ'] += -0.5*np.einsum('JNFB,MNEF->MBEJ', self.T2['IJAB'], self.V('MNEF'))
        self.W['MBEJ'] += 0.5*np.einsum('JnBf,MnEf->MBEJ', self.T2['IjAb'], self.V('MnEf'))

        # Update W(mbej)
        self.W.update({'mbej' : np.zeros((self.nbeta, self.bvir, self.bvir, self.nbeta))})
        self.W['mbej'] += self.V('mbej')
        self.W['mbej'] += -0.5*np.einsum('jnfb,mnef->mbej', self.T2['ijab'], self.V('mnef'))
        self.W['mbej'] += 0.5*np.einsum('NjFb,mNeF->mbej', self.T2['IjAb'], self.V('mNeF'))

        # Update W(MbEj)
        self.W.update({'MbEj' : np.zeros((self.nalpha, self.bvir, self.avir, self.nbeta))})
        self.W['MbEj'] += self.V('MbEj')
        self.W['MbEj'] += -0.5*np.einsum('jnfb,MnEf->MbEj', self.T2['ijab'], self.V('MnEf'))
        self.W['MbEj'] += 0.5*np.einsum('NjFb,MNEF->MbEj', self.T2['IjAb'], self.V('MNEF'))

        # Update W(MbeJ)
        self.W.update({'MbeJ' : np.zeros((self.nalpha, self.bvir, self.bvir, self.nalpha))})
        self.W['MbeJ'] += self.V('MbeJ')
        self.W['MbeJ'] += -0.5*np.einsum('JnFb,MneF->MbeJ', self.T2['IjAb'], self.V('MneF'))

        # Update W(mBeJ)
        self.W.update({'mBeJ' : np.zeros((self.nbeta, self.avir, self.bvir, self.nalpha))})
        self.W['mBeJ'] += self.V('mBeJ')
        self.W['mBeJ'] += -0.5*np.einsum('JNFB,mNeF->mBeJ', self.T2['IJAB'], self.V('mNeF'))
        self.W['mBeJ'] += 0.5*np.einsum('JnBf,mnef->mBeJ', self.T2['IjAb'], self.V('mnef'))

        # Update W(mBEj)
        self.W.update({'mBEj' : np.zeros((self.nbeta, self.avir, self.avir, self.nbeta))})
        self.W['mBEj'] += self.V('mBEj')
        self.W['mBEj'] += -0.5*np.einsum('NjBf,mNEf->mBEj', self.T2['IjAb'], self.V('mNEf'))

    def update_amp(self):

        # Create a new set of amplitudes

        newT2 = {
        'IJAB' : np.zeros(self.T2['IJAB'].shape),
        'ijab' : np.zeros(self.T2['ijab'].shape),
        'IjAb' : np.zeros(self.T2['IjAb'].shape),
        }

        # Update T(IJAB)

        newT2['IJAB'] += self.V('IJAB')
        newT2['IJAB'] += 0.5*np.einsum('MNAB,MNIJ->IJAB', self.T2['IJAB'], self.W['MNIJ'])
        newT2['IJAB'] += 0.5*np.einsum('IJEF,ABEF->IJAB', self.T2['IJAB'], self.W['ABEF'])

        Pab =  np.einsum('IJAE,BE->IJAB', self.T2['IJAB'], self.F['AE']) 
        Pij = -np.einsum('IMAB,MJ->IJAB', self.T2['IJAB'], self.F['MI']) 

        Pijab  =  np.einsum('IMAE,MBEJ->IJAB', self.T2['IJAB'], self.W['MBEJ'])
        Pijab +=  np.einsum('ImAe,mBeJ->IJAB', self.T2['IjAb'], self.W['mBeJ'])

        newT2['IJAB'] += Pab + Pij + Pijab - (Pab + Pijab).transpose(0,1,3,2) - (Pij + Pijab).transpose(1,0,2,3) + Pijab.transpose(1,0,3,2)
         
        newT2['IJAB'] *= self.D['IJAB']

        # Update T(ijab)

        newT2['ijab'] += self.V('ijab')
        newT2['ijab'] += 0.5*np.einsum('mnab,mnij->ijab', self.T2['ijab'], self.W['mnij'])
        newT2['ijab'] += 0.5*np.einsum('ijef,abef->ijab', self.T2['ijab'], self.W['abef'])

        Pab = np.einsum('ijae,be->ijab', self.T2['ijab'], self.F['ae']) 
        Pij = -np.einsum('imab,mj->ijab', self.T2['ijab'], self.F['mi']) 

        Pijab  =  np.einsum('imae,mbej->ijab', self.T2['ijab'], self.W['mbej'])
        Pijab += np.einsum('MiEa,MbEj->ijab', self.T2['IjAb'], self.W['MbEj'])

        newT2['ijab'] += Pab + Pij + Pijab - (Pab + Pijab).transpose(0,1,3,2) - (Pij + Pijab).transpose(1,0,2,3) + Pijab.transpose(1,0,3,2)

        newT2['ijab'] *= self.D['ijab']

        # Update T(IjAb)

        newT2['IjAb'] += self.V('IjAb')

        newT2['IjAb'] +=  np.einsum('IjAe,be->IjAb', self.T2['IjAb'], self.F['ae'])
        newT2['IjAb'] +=  np.einsum('IjEb,AE->IjAb', self.T2['IjAb'], self.F['AE'])
        newT2['IjAb'] += -np.einsum('ImAb,mj->IjAb', self.T2['IjAb'], self.F['mi'])
        newT2['IjAb'] += -np.einsum('MjAb,MI->IjAb', self.T2['IjAb'], self.F['MI'])

        newT2['IjAb'] +=  np.einsum('MnAb,MnIj->IjAb', self.T2['IjAb'], self.W['MnIj'])
        newT2['IjAb'] +=  np.einsum('IjEf,AbEf->IjAb', self.T2['IjAb'], self.W['AbEf'])

        newT2['IjAb'] +=  np.einsum('IMAE,MbEj->IjAb', self.T2['IJAB'], self.W['MbEj'])

        newT2['IjAb'] +=  np.einsum('ImAe,mbej->IjAb', self.T2['IjAb'], self.W['mbej'])
        
        newT2['IjAb'] +=  np.einsum('ImEb,mAEj->IjAb', self.T2['IjAb'], self.W['mBEj'])

        newT2['IjAb'] +=  np.einsum('MjAe,MbeI->IjAb', self.T2['IjAb'], self.W['MbeJ'])

        newT2['IjAb'] +=  np.einsum('jmbe,mAeI->IjAb', self.T2['ijab'], self.W['mBeJ'])

        newT2['IjAb'] +=  np.einsum('MjEb,MAEI->IjAb', self.T2['IjAb'], self.W['MBEJ'])

        newT2['IjAb'] *= self.D['IjAb']

        # Compute RMS

        self.rms['IJAB'] = np.sqrt(np.sum(np.square(newT2['IJAB'] - self.T2['IJAB'] )))/(self.nalpha*self.nalpha*self.avir*self.avir)
        self.rms['ijab'] = np.sqrt(np.sum(np.square(newT2['ijab'] - self.T2['ijab'] )))/(self.nbeta*self.nbeta*self.bvir*self.bvir)
        self.rms['IjAb'] = np.sqrt(np.sum(np.square(newT2['IjAb'] - self.T2['IjAb'] )))/(self.nalpha*self.nbeta*self.avir*self.bvir)

        # Save new amplitudes

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

        # Save two-electron integral in physicists notation (3 Spin cases)
        self.Vaaaa = Vaaaa.swapaxes(1,2)
        self.Vaaaa = self.Vaaaa - self.Vaaaa.swapaxes(2,3)

        self.Vbbbb = Vbbbb.swapaxes(1,2)
        self.Vbbbb = self.Vbbbb - self.Vbbbb.swapaxes(2,3)

        self.Vabab = Vaabb.swapaxes(1,2)

        self.compute()

    def compute(self):

        # Auxiliar D matrices

        new = np.newaxis

        self.D = {
        'IJAB' : 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :]),
        'ijab' : 1.0/(self.fock_od[:, new, new, new] + self.fock_od[new, :, new, new] - self.fock_vd[new, new, :, new] - self.fock_vd[new, new, new, :]),
        'IjAb' : 1.0/(self.fock_Od[:, new, new, new] + self.fock_od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_vd[new, new, new, :]),
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


        # Initial T2 amplitudes

        self.rms = {
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
            max_rms = self.rms[max(self.rms)]
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {:< 15.10f}".format(self.Ecc))
            print("Energy change:         {:< 15.10f}".format(dE))
            print("Max RMS residue:       {:< 15.10f}".format(max_rms))
            print("Time required:         {:< 15.10f}".format(time.time() - t))
            print('='*37)
            ite += 1

        print('CC Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))
        print('CCD iterations took %.2f seconds.\n' % (time.time() - t0))
