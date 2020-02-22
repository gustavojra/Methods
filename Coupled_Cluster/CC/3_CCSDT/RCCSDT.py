import psi4
import os
import sys
import numpy as np
import time
import copy

sys.path.append('../../Aux')
from tools import *

np.set_printoptions(suppress=True, linewidth=120)

class RCCSDT:

    def update_energy(self):
        
        X = 2*self.T2 + 2*np.einsum('IA,JB->IJAB', self.T1, self.T1,optimize='optimal')
        X += - self.T2.transpose(1,0,2,3) - np.einsum('JA,IB->IJAB', self.T1, self.T1,optimize='optimal')
        self.Ecc = np.einsum('IjAb,IjAb->', X, self.Voovv,optimize='optimal')

    def build_Fint(self):
        # alternative form

        Fme = self.fock_OV + np.einsum('nmfe,nf->me', self.Aoovv, self.T1, optimize='optimal')

        Fmi = self.fock_OO + np.einsum('me,ie->mi', Fme, self.T1, optimize='optimal')
        Fmi += np.einsum('nmfe,nife->mi', self.Aoovv, self.T2, optimize='optimal')
        Fmi += np.einsum('mnif,nf->mi', self.Aooov, self.T1, optimize='optimal')

        Fae = self.fock_VV - np.einsum('me,ma->ae', Fme, self.T1, optimize='optimal')
        Fae += -np.einsum('nmfe,nmfa->ae', self.Aoovv, self.T2, optimize='optimal')
        Fae += np.einsum('mafe,mf->ae', self.Aovvv, self.T1, optimize='optimal')

        return Fae, Fmi, Fme

    def get_Fint(self):

        tau = self.T2 + 0.5*np.einsum('IA,JB->IJAB', self.T1, self.T1,optimize='optimal')

        # Compute F(AE)
        Fae = self.fock_VV - 0.5*np.einsum('ME,MA->AE', self.fock_OV, self.T1,optimize='optimal')
        Fae += np.einsum('MF,MAFE->AE', self.T1, self.Aovvv,optimize='optimal')
        Fae += -np.einsum('MnAf,MnEf->AE', tau, self.Aoovv,optimize='optimal')

        # Compute F(MI)
        Fmi = self.fock_OO + 0.5*np.einsum('ME,IE->MI', self.fock_OV, self.T1,optimize='optimal')
        Fmi += np.einsum('NE,MNIE->MI', self.T1, self.Aooov,optimize='optimal')
        Fmi += np.einsum('INEF,MNEF->MI', tau, self.Aoovv,optimize='optimal')

        # Compute F(ME)
        Fme = self.fock_OV + np.einsum('NF, MNEF-> ME', self.T1, self.Aoovv,optimize='optimal')

        return Fae, Fmi, Fme

    def get_Winf(self, Te):

        # Compute W(MnIj)
        Wmnij = np.einsum('je, MnIe-> MnIj', self.T1, self.Vooov,optimize='optimal')
        Wmnij += self.Voooo
        Wmnij += np.einsum('IE,nMjE -> MnIj', self.T1, self.Vooov,optimize='optimal')
        Wmnij += (1.0/2.0)*np.einsum('IjEf,MnEf->MnIj', Te, self.Voovv,optimize='optimal')

        # Compute W(AbEf)
        Wabef = -np.einsum('mb, mAfE-> AbEf', self.T1, self.Vovvv,optimize='optimal')
        Wabef += self.Vvvvv
        Wabef += -np.einsum('MA, MbEf -> AbEf', self.T1, self.Vovvv,optimize='optimal')
        Wabef += (1.0/2.0)*np.einsum('MnAb,MnEf->AbEf', Te, self.Voovv,optimize='optimal')

        # Compute W(MbEj)
        W_MbEj = np.einsum('jf,MbEf->MbEj', self.T1, self.Vovvv,optimize='optimal')
        W_MbEj += self.Voovv.transpose(0,3,2,1)
        W_MbEj += -np.einsum('nb,nMjE->MbEj', self.T1, self.Vooov,optimize='optimal')
        X = self.T2 + 2*np.einsum('jf,nb->jnfb', self.T1, self.T1,optimize='optimal')
        W_MbEj += -0.5*np.einsum('jnfb,MnEf->MbEj', X, self.Voovv,optimize='optimal')
        W_MbEj += 0.5*np.einsum('NjFb,MNEF->MbEj', self.T2, self.Aoovv,optimize='optimal')

        # Compute W(MbeJ)
        W_MbeJ = -self.Vovov.transpose(0,1,3,2)
        W_MbeJ += -np.einsum('JF,MbFe->MbeJ', self.T1, self.Vovvv,optimize='optimal')
        W_MbeJ += np.einsum('nb,MnJe->MbeJ', self.T1, self.Vooov,optimize='optimal')
        X = 0.5*self.T2 + np.einsum('JF,nb->JnFb', self.T1, self.T1,optimize='optimal')
        W_MbeJ += np.einsum('JnFb,nMeF->MbeJ', X, self.Voovv,optimize='optimal')

        return Wmnij, Wabef, W_MbEj, W_MbeJ

    def update_amp(self):

        # Slices
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nmo)

        Te = self.T2 + np.einsum('IA,JB->IJAB', self.T1, self.T1,optimize='optimal')
        Fae, Fmi, Fme = self.get_Fint()
        Wmnij, Wabef, W_MbEj, W_MbeJ = self.get_Winf(Te)

        # NEW INTERMEDIATES
        altFae, altFmi, altFme = self.build_Fint()

        vWefam = self.Vovvv.transpose(3,2,1,0) - np.einsum('nmef,na->efam', self.Voovv, self.T1)

        vWejmn = self.Vooov.transpose(3,2,1,0) + np.einsum('mnef,jf->ejmn', self.Voovv, self.T1)

        tvWijam = self.Vgeneral[o,o,v,o] + np.einsum('maje,ie->ijam', self.Vovov, self.T1, optimize='optimal')
        #tvWijam = self.Vooov.transpose(0,1,3,2) + np.einsum('maje,ie->ijam', self.Vovov, self.T1, optimize='optimal')
        tvWijam += np.einsum('imae,je->ijam', self.Voovv, self.T1, optimize='optimal')
        tvWijam += np.einsum('mafe,ijef->ijam', self.Vovvv, Te, optimize='optimal')

        tvWeima = self.Vgeneral[v,o,o,v] - np.einsum('eimn,na->eima', vWejmn, self.T1, optimize='optimal')
        #tvWeima = self.Voovv.transpose(2,1,0,3) - np.einsum('eimn,na->eima', vWejmn, self.T1, optimize='optimal')
        tvWeima += np.einsum('maef,if->eima', self.Vovvv, self.T1, optimize='optimal')
        tvWeima += (1.0/4.0)*np.einsum('nmfe,nifa->eima', self.Aoovv, 2*self.T2 - self.T2.transpose(0,1,3,2), optimize='optimal')
        tvWeima += -(1.0/4.0)*np.einsum('nmef,infa->eima', self.Voovv, self.T2, optimize='optimal')

        tvWiema = self.Vovov - np.einsum('eimn,na->iema', vWejmn, self.T1, optimize='optimal') ## weird
        tvWiema += np.einsum('mafe,if->iema', self.Vovvv, self.T1, optimize='optimal')
        tvWiema += -0.5*np.einsum('nmef,infa->iema', self.Voovv, self.T2, optimize='optimal')

        vWijmn = 0.5*self.Voooo + np.einsum('mnie,je->ijmn', self.Vooov, self.T1, optimize='optimal')
        vWijmn += 0.5*np.einsum('mnef,ijef->ijmn', self.Voovv, Te, optimize='optimal')
        vWijmn += vWijmn.transpose(1,0,2,3)

        # Create a new set of amplitudes
        newT1 = np.zeros(self.T1.shape)
        newT2 = np.zeros(self.T2.shape)

        # Update T(IA)

        newT1 += self.fock_OV
        newT1 += np.einsum('ea,ie->ia', self.fock_VV, self.T1, optimize='optimal')
        newT1 += -np.einsum('mi,ma->ia', altFmi, self.T1, optimize='optimal')
        newT1 += np.einsum('maei,me->ia', self.tempAovvo, self.T1, optimize='optimal')
        newT1 += np.einsum('me,miea->ia', altFme, 2*self.T2-self.T2.swapaxes(2,3), optimize='optimal')
        newT1 += np.einsum('maef,mief->ia', self.Vovvv, 2*Te-Te.swapaxes(2,3), optimize='optimal')
        newT1 += -np.einsum('eimn,mnea->ia', 2*vWejmn-vWejmn.swapaxes(2,3), self.T2, optimize='optimal')

        # T1 <- T3(IjKAbC)
        np.einsum('ijab, jiupab -> up', self.Aoovv, self.T3)

        newT1 *= self.d

        # Update T(IjAb)

        newerT2 = 0.5*self.Voovv
        newerT2 += np.einsum('ieab,je->ijab', self.Vovvv, self.T1, optimize='optimal')
        newerT2 += -np.einsum('ijam,mb->ijab', tvWijam, self.T1, optimize='optimal')
        newerT2 += np.einsum('ae,ijeb->ijab', altFae, self.T2, optimize='optimal')
        newerT2 += -np.einsum('mi,mjab->ijab', altFmi, self.T2, optimize='optimal')
        # not sure about this mess
        #X = tvWeima.transpose(2,1,0,3)
        #newerT2 += 0.5*np.einsum('miea,mjeb->ijab', 2*X-X.swapaxes(2,3), 2*self.T2-self.T2.swapaxes(2,3), optimize='optimal')
        ###

        newerT2 += 2*np.einsum('eima,mjeb->ijab', tvWeima, self.T2, optimize='optimal')
        newerT2 += -np.einsum('eima,mjbe->ijab', tvWeima, self.T2, optimize='optimal')
        newerT2 += -np.einsum('iema,mjeb->ijab', tvWiema, self.T2, optimize='optimal')
        newerT2 += 0.5*np.einsum('iema,mjbe->ijab', tvWiema, self.T2, optimize='optimal')

        X = np.einsum('iema,jmeb->ijab', tvWiema, self.T2, optimize='optimal')
        newerT2 -= 0.5*X + X.transpose(0,1,3,2)

        newerT2 += 0.5*np.einsum('ijmn,mnab->ijab', vWijmn, Te, optimize='optimal')
        newerT2 += 0.5*np.einsum('abef,ijef->ijab', self.Vvvvv, Te, optimize='optimal')

        newerT2 = (newerT2 + newerT2.transpose(1,0,3,2))*self.D

        ###
        #newT2 += self.Voovv
        #newT2 += np.einsum('mnab,mnij->ijab', Te, Wmnij,optimize='optimal')
        #newT2 += np.einsum('ijef,abef->ijab', Te, Wabef,optimize='optimal')

        #X = Fae - 0.5*np.einsum('mb,me->be', self.T1, Fme,optimize='optimal')
        #P = np.einsum('IjAe,be->IjAb', self.T2, X,optimize='optimal')
        #X = Fmi + 0.5*np.einsum('je,me->mj', self.T1, Fme,optimize='optimal')
        #P += - np.einsum('ImAb,mj->IjAb', self.T2, X,optimize='optimal')
        #
        #X = self.T2 - self.T2.transpose(1,0,2,3)
        #P += np.einsum('imae,mbej->ijab', X, W_MbEj,optimize='optimal')
        #P += -np.einsum('ie,ma,mbej->ijab', self.T1, self.T1, self.Voovv.transpose(0,3,2,1),optimize='optimal')

        #P += np.einsum('imae,mbej->ijab', self.T2, W_MbEj + W_MbeJ,optimize='optimal')

        #P += np.einsum('mibe,maej->ijab', self.T2, W_MbeJ,optimize='optimal')
        #P += -np.einsum('ie,mb,maje->ijab', self.T1, self.T1, self.Vovov,optimize='optimal')

        #P += np.einsum('IE,jEbA->IjAb', self.T1, self.Vovvv,optimize='optimal')
        #P += -np.einsum('MA,IjMb->IjAb', self.T1, self.Vooov,optimize='optimal')

        ##newT2 = newT2 + newT2.transpose(1,0,3,2)
        #newT2 += P + P.transpose(1,0,3,2)

        #newT2 *= self.D
        newT2 = newerT2

        # Update T3

        # Compute RMS

        self.rms1 = np.sqrt(np.sum(np.square(newT1 - self.T1 )))/(self.ndocc*self.nvir)
        self.rms2 = np.sqrt(np.sum(np.square(newT2 - self.T2 )))/(self.ndocc*self.ndocc*self.nvir*self.nvir)
        print('NewxNewer: ',np.sqrt(np.sum(np.square(newT2 - newerT2)))/(self.ndocc*self.ndocc*self.nvir*self.nvir))
        # Save new amplitudes

        self.T1 = newT1
        self.T2 = newT2

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):

        # Save reference wavefunction properties

        print('RUNNING CCSDT')

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

        # Save slices of two-electron repulsion integral
        Vphys = Vchem.swapaxes(1,2)
        Vsa = 2*Vphys - Vphys.swapaxes(2,3)

        self.Aovvv = Vsa[o,v,v,v]
        self.Aooov = Vsa[o,o,o,v]
        self.Aoovv = Vsa[o,o,v,v]
        self.Avoov = Vsa[v,o,o,v]
        self.tempAovvo = Vsa[o,v,v,o]

        self.Voooo = Vphys[o,o,o,o]
        self.Vooov = Vphys[o,o,o,v]
        self.Voovv = Vphys[o,o,v,v]
        self.Vovov = Vphys[o,v,o,v]
        self.Vovvv = Vphys[o,v,v,v]
        self.Vvvvv = Vphys[v,v,v,v]
        self.tempVvooo = Vphys[v,o,o,o]
        self.Vgeneral = Vphys

        self.compute()

    def compute(self):

        # Auxiliar D matrices

        new = np.newaxis
        self.d = 1.0/(self.fock_Od[:, new] - self.fock_Vd[new, :])

        self.D = 1.0/(self.fock_Od[:, new, new, new] + self.fock_Od[new, :, new, new] - self.fock_Vd[new, new, :, new] - self.fock_Vd[new, new, new, :])

        self.Dd = 1.0/(self.fock_Od[:, new, new, new, new, new] + self.fock_Od[new, :, new, new, new, new] + self.fock_Od[new, new, :, new, new, new]\
                     - self.fock_Vd[new, new, new, :, new, new] - self.fock_Vd[new, new, new, new, :, new] - self.fock_Vd[new, new, new, new, new, :])

        # Initial T1 amplitudes

        self.T1 = self.fock_OV*self.d

        # Initial T2 amplitudes

        self.T2 = self.D*self.Voovv

        # Initial T3 amplitudes

        self.T3 = np.zeros(self.Dd.shape)

        # Get MP2 energy

        self.update_energy()

        print('MP2 Energy:   {:<15.10f}'.format(self.Ecc + self.Ehf))

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
        print('CCSDT iterations took %.2f seconds.\n' % (time.time() - t0))
