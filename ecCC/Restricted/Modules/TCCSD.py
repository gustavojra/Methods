import psi4
import os
import sys
import numpy as np
import time
import copy
import re

sys.path.append('./')

from CASDecom import CASDecom
from Det import Det

class TCCSD:

    def __init__(self, wfn, CC_CONV=6, E_CONV = 8, CC_MAXITER = 50):

        # Collect data from CI wavefunction

        self.Escf = wfn.energy()
        self.nelec = wfn.nalpha() + wfn.nbeta()
        if self.nelec % 2 != 0:
            raise NameError('Number of electrons cannot be odd for RHF')
        self.C = wfn.Ca()
        self.ndocc = int(self.nelec/2)
        self.nmo = wfn.nmo()
        self.nvir = self.nmo - self.ndocc
        self.eps = np.asarray(wfn.epsilon_a())
        self.nbf = self.C.shape[0]
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()
        self.fdocc = sum(wfn.frzcpi())
        self.fvir = sum(wfn.frzvpi())
        
        print("Number of Electrons:            {}".format(self.nelec))
        print("Number of Basis Functions:      {}".format(self.nbf))
        print("Number of Molecular Orbitals:   {}".format(self.nmo))
        print("Number of Doubly ocuppied MOs:  {}\n".format(self.ndocc))
        print("Number of Frozen dobly occ MOs: {}\n".format(self.fdocc))
        print("Number of Frozen virtual MOs:   {}\n".format(self.fvir))

        # Build integrals from Psi4 MINTS
    
        print("Converting atomic integrals to MO integrals...")
        t = time.time()
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.Vint = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
        self.Vint = self.Vint.swapaxes(1,2) # Convert to Physicists' notation
        self.h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.h = np.einsum('up,vq,uv->pq', self.C, self.C, self.h)
        print("Completed in {} seconds!".format(time.time()-t))

        self.compute(CC_CONV=CC_CONV, E_CONV=E_CONV, CC_MAXITER=CC_MAXITER)

    def cc_energy(self):
    
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)
        tau = self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1)
        X = 2*tau - np.einsum('ijab->jiab',tau)
        self.Ecc = np.einsum('abij,ijab->', self.Vint[v,v,o,o], X)

    def T1_T2_Update(self, EINSUMOPT='optimal'):
    
        # Compute CCSD Amplitudes. Only the T1 (alpha -> alpha) are considered since the beta -> beta case yields the same amplitude and the mixed case is zero.
        # For T2 amplitudes we consider the case (alpha -> alpha, beta -> beta) the other spin cases can be writen in terms of this one.
        # Equations from J. Chem. Phys. 86, 2881 (1987): G. E. Scuseria et al.
        # CC Intermediate arrays
    
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        tau = self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1,optimize=EINSUMOPT)
        Te = 0.5*self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1,optimize=EINSUMOPT)
    
        A2l = np.einsum('uvij,ijpg->uvpg', self.Vint[o,o,o,o], tau,                                    optimize=EINSUMOPT)
        B2l = np.einsum('abpg,uvab->uvpg', self.Vint[v,v,v,v], tau,                                    optimize=EINSUMOPT)
        C1  = np.einsum('uaip,ia->uip',    self.Vint[o,v,o,v], self.T1,                                optimize=EINSUMOPT) 
        C2  = np.einsum('aupi,viga->pvug', self.Vint[v,o,v,o], self.T2,                                optimize=EINSUMOPT)
        C2l = np.einsum('iaug,ivpa->pvug', self.Vint[o,v,o,v], tau,                                    optimize=EINSUMOPT)
        D1  = np.einsum('uapi,va->uvpi',   self.Vint[o,v,v,o], self.T1,                                optimize=EINSUMOPT)
        D2l = np.einsum('abij,uvab->uvij', self.Vint[v,v,o,o], tau,                                    optimize=EINSUMOPT)
        Ds2l= np.einsum('acij,ijpb->acpb', self.Vint[v,v,o,o], tau,                                    optimize=EINSUMOPT)
        D2a = np.einsum('baji,vjgb->avig', self.Vint[v,v,o,o], 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT)
        D2b = np.einsum('baij,vjgb->avig', self.Vint[v,v,o,o], self.T2,                                optimize=EINSUMOPT)
        D2c = np.einsum('baij,vjbg->avig', self.Vint[v,v,o,o], self.T2,                                optimize=EINSUMOPT)
        Es1 = np.einsum('uvpi,ig->uvpg',   self.Vint[o,o,v,o], self.T1,                                optimize=EINSUMOPT)
        E1  = np.einsum('uaij,va->uvij',   self.Vint[o,v,o,o], self.T1,                                optimize=EINSUMOPT)
        E2a = np.einsum('buji,vjgb->uvig', self.Vint[v,o,o,o], 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT)
        E2b = np.einsum('buij,vjgb->uvig', self.Vint[v,o,o,o], self.T2,                                optimize=EINSUMOPT)
        E2c = np.einsum('buij,vjbg->uvig', self.Vint[v,o,o,o], self.T2,                                optimize=EINSUMOPT)
        F11 = np.einsum('bapi,va->bvpi',   self.Vint[v,v,v,o], self.T1,                                optimize=EINSUMOPT)
        F12 = np.einsum('baip,va->bvip',   self.Vint[v,v,o,v], self.T1,                                optimize=EINSUMOPT)
        Fs1 = np.einsum('acpi,ib->acpb',   self.Vint[v,v,v,o], self.T1,                                optimize=EINSUMOPT)
        F2a = np.einsum('abpi,uiab->aup',  self.Vint[v,v,v,o], 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT) 
        F2l = np.einsum('abpi,uvab->uvpi', self.Vint[v,v,v,o], tau,                                    optimize=EINSUMOPT)
    
        X = E1 + D2l

        giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2), optimize=EINSUMOPT)
        
        X = Fs1 - Ds2l
        gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3), optimize=EINSUMOPT)
    
        # T2 Amplitudes update
    
        J = np.einsum('ag,uvpa->uvpg', gap, self.T2, optimize=EINSUMOPT) - np.einsum('vi,uipg->uvpg', giu, self.T2, optimize=EINSUMOPT)
    
        S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  
        S +=     np.einsum('avig,uipa->uvpg', (D2a-D2b), self.T2 - Te.transpose(0,1,3,2),  optimize=EINSUMOPT)
        S += 0.5*np.einsum('avig,uipa->uvpg', D2c, self.T2,                                optimize=EINSUMOPT)
        S +=     np.einsum('auig,viap->uvpg', D2c, Te,                                     optimize=EINSUMOPT)
        S +=     np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau,                           optimize=EINSUMOPT)
        S -=     np.einsum('uvpi,ig->uvpg',   D1 + F2l, self.T1,                           optimize=EINSUMOPT)
        S -=     np.einsum('uvig,ip->uvpg',   E2a - E2b - E2c.transpose(1,0,2,3), self.T1, optimize=EINSUMOPT)
        S -=     np.einsum('avgi,uipa->uvpg', F11, self.T2,                                optimize=EINSUMOPT)
        S -=     np.einsum('avpi,uiag->uvpg', F11, self.T2,                                optimize=EINSUMOPT)
        S +=     np.einsum('avig,uipa->uvpg', F12, 2*self.T2 - self.T2.transpose(0,1,3,2), optimize=EINSUMOPT)
    
        T2new = self.Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)
    
        T2new = np.einsum('uvpg,uvpg->uvpg', T2new, self.D,optimize=EINSUMOPT)

    
        #self.r2 = np.sum(np.abs(T2new - self.T2)) #- np.sum(np.abs(self.internal_T2))
    
        # T1 Amplitudes update
        
        T1new =    np.einsum('ui,ip->up',      giu, self.T1,                                   optimize=EINSUMOPT)
        T1new -=   np.einsum('ap,ua->up',      gap, self.T1,                                   optimize=EINSUMOPT)
        T1new -=   np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), self.T1, self.T1, optimize=EINSUMOPT)
        T1new -=   np.einsum('auip,ia->up',    2*(D2a - D2b) + D2c, self.T1,                   optimize=EINSUMOPT)
        T1new -=   np.einsum('aup->up',        F2a,                                            optimize=EINSUMOPT)
        T1new +=   np.einsum('uiip->up',       1.0/2.0*(E2a - E2b) + E2c,                      optimize=EINSUMOPT)
        T1new +=   np.einsum('uip->up',        C1,                                             optimize=EINSUMOPT)
        T1new -= 2*np.einsum('uipi->up',       D1,                                             optimize=EINSUMOPT)
    
        T1new = np.einsum('up,up->up', T1new, self.d, optimize=EINSUMOPT)

        T1new[self.h_space, self.p_space] = self.FROZEN_T1
        T2new[self.h_space, self.h_space, self.p_space, self.p_space] = self.FROZEN_T2

        self.r1 = np.sum(np.sqrt(np.square(T1new - self.T1)))/(self.nvir*self.ndocc)
        self.r2 = np.sum(np.sqrt(np.square(T2new - self.T2)))/((self.nvir*self.ndocc)**2)

        self.T1 = T1new
        self.T2 = T2new
        

    def compute(self, CC_CONV=6, E_CONV = 8, CC_MAXITER=50):
        
        ############### STEP 1 ###############
        ##############  DETCI  ###############

        # Retrive CI coefficients and determinant objects from DETCI

        self.ref = Det(a = '1'*self.ndocc + '0'*self.nvir, b = '1'*self.ndocc + '0'*self.nvir)

        pattern = '\s*?\*\s+?\d+?\s+?([-\s]\d\.\d+?)\s+?\(.+?\)\s+?(.+?\n)'
        self.Ccas = []
        self.determinants = []
        dets_string = []
        with open('output.dat', 'r') as output:
            for line in output:
                m = re.match(pattern, line)
                if m:
                    self.Ccas.append(float(m.group(1)))
                    dets_string.append(m.group(2))
        
        for det in dets_string:
            #print('Translating: {}'.format(det))
            a_index = []
            b_index = []
            for o in det.split():
               # print('Checking piece {}'.format(o))
                if o[-1] == 'X' or o[-1] == 'A':
                    a_index.append(int(o[:-2])-1)
                 #   print('alpha occupied')
                if o[-1] == 'X' or o[-1] == 'B':
                    b_index.append(int(o[:-2])-1)
                #    print('beta occupied')
            #print(a_index)
            #print(b_index)
            a_string = '1'*self.fdocc
            b_string = '1'*self.fdocc
            #print(a_string)
            #print(b_string)
            for i in range(self.fdocc,self.nmo):
                if i in a_index:
                    a_string += '1'
                else:
                    a_string += '0'
                if i in b_index:
                    b_string += '1'
                else:
                    b_string += '0'
            self.determinants.append(Det(a = a_string, b = b_string, ref = self.ref, sq = True))
        for i,d in enumerate(self.determinants):
            #print(d)
            self.Ccas[i] *= d.order
        
        ############### STEP 2 ###############
        ############  CASDecom  ##############

        # Run CASDecom to translate CI coefficients into CC amplitudes

        self.FROZEN_T1, self.FROZEN_T2 = \
        CASDecom(self.Ccas, self.determinants, self.ref, fdocc = self.fdocc, fvir = self.fvir, return_t3=False, return_t4=False)

        self.h_space = slice(self.fdocc, self.ndocc)
        self.p_space = slice(0, self.nvir-self.fvir)

        self.FROZEN_T1 = self.FROZEN_T1[self.h_space, self.p_space]
        self.FROZEN_T2 = self.FROZEN_T2[self.h_space, self.h_space, self.p_space, self.p_space]


       # Compute CCSD 

        # Slices
        
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        # START CCSD CODE

        # Build the Auxiliar Matrix D

        print('Building Auxiliar D matrix...\n')
        t = time.time()
        self.D  = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])
        self.d  = np.zeros([self.ndocc, self.nvir])
        
        for i,ei in enumerate(self.eps[o]):
            for j,ej in enumerate(self.eps[o]):
                for a,ea in enumerate(self.eps[v]):
                    self.d[i,a] = 1/(ea - ei)
                    for b,eb in enumerate(self.eps[v]):
                        self.D[i,j,a,b] = 1/(ei + ej - ea - eb)

        print('Done. Time required: {:.5f} seconds'.format(time.time() - t))
        t = time.time()
        
        # Generate initial T1 and T2 amplitudes

        self.T1 = np.zeros([self.ndocc, self.nvir])
        self.T2  = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])
        
        self.T1[self.h_space, self.p_space] = self.FROZEN_T1
        self.T2[self.h_space, self.h_space, self.p_space, self.p_space] = self.FROZEN_T2

        self.cc_energy()

        print('CC Energy from CAS Amplitudes: {:<5.10f}'.format(self.Ecc+self.Escf))

        self.r1 = 1
        self.r2 = 1
            
        LIM = 10**(-CC_CONV)
        ELIM = 10**(-E_CONV)
        ite = 0
        
        while self.r2 > LIM or self.r1 > LIM or abs(dE) > ELIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            Eold = self.Ecc
            t = time.time()
            self.T1_T2_Update()
            self.cc_energy()
            dE = self.Ecc - Eold
            print("Iteration {}".format(ite))
            print("Correlation energy:    {:<5.10f}".format(self.Ecc))
            print("Energy change:         {:<5.10f}".format(dE))
            print("T1 Residue:            {:>13.2E}".format(self.r1))
            print("T2 Residue:            {:>13.2E}".format(self.r2))
            print("Time required (s):     {:< 5.10f}".format(time.time() - t))
            print('='*36)
        self.Ecc = self.Ecc + self.Escf

        print("\nTCC Equations Converged!!!")
        print("Final TCCSD Energy:     {:<5.10f}".format(self.Ecc))

        
        
