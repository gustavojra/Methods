import psi4
import sys
import numpy as np
import time
import copy
import re

sys.path.append('./')

from CASDecom import CASDecom
from Det import Det
from CIfromDETCI import read_ci_vec
from printtensor import *
from save_amp import *

print_to_output = True

def printout(x):
    if print_to_output:
        psi4.core.print_out(x)
        psi4.core.print_out('\n')
    else:
        print(x)

class CASCCSD:

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV = 8, MP2_GUESS=False, RELAX_T3T1=True, read_amp=False, write_amp=False, fcore=False):

        # Collect data from CI wavefunction

        self.Escf = wfn.reference_wavefunction().energy()
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
        self.read_amp = read_amp
        self.write_amp = write_amp
        self.fcore = fcore
        
        printout(\
        """---------------------------------------------------------
                          CASCCSD STARTED
        ---------------------------------------------------------""")
        printout("Number of Electrons:            {}".format(self.nelec))
        printout("Number of Basis Functions:      {}".format(self.nbf))
        printout("Number of Molecular Orbitals:   {}".format(self.nmo))
        printout("Number of Doubly ocuppied MOs:  {}\n".format(self.ndocc))
        printout("Number of Frozen dobly occ MOs: {}\n".format(self.fdocc))
        printout("Number of Frozen virtual MOs:   {}\n".format(self.fvir))

        # Build integrals from Psi4 MINTS
    
        printout("Converting atomic integrals to MO integrals...")
        t = time.time()
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.Vint = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
        self.Vint = self.Vint.swapaxes(1,2) # Convert to Physicists' notation
        self.h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.h = np.einsum('up,vq,uv->pq', self.C, self.C, self.h)
        printout("Completed in {} seconds!".format(time.time()-t))

        self.compute(CC_CONV, CC_MAXITER, E_CONV, MP2_GUESS, RELAX_T3T1)

    def cc_energy(self):
        
        # Compute the Coupled Cluster energy given T1 and T2 amplitudes
        # Equation from J. Chem. Phys. 86, 2881 (1987): G. E. Scuseria et al.
    
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        tau = self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1)
        X = 2*tau - np.einsum('ijab->jiab',tau)
        self.Ecc = np.einsum('abij,ijab->', self.Vint[v,v,o,o], X)

    def T1_T2_Update(self, RELAX_T3T1 = True, EINSUMOPT='optimal'):
    
        # Compute CCSD Amplitudes. Only the T1 (alpha -> alpha) are considered since the beta -> beta case yields the same amplitude and the mixed case is zero.
        # For T2 amplitudes we consider the case (alpha -> alpha, beta -> beta) the other spin cases can be writen in terms of this one.
        # Equations from J. Chem. Phys. 86, 2881 (1987): G. E. Scuseria et al.

        ## CC Intermediate arrays
    
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
        
        ## Update the contribution of T3 on T2 via the disconnected T3*T1 term. This is true by default. 

        if RELAX_T3T1:
            self.relax_t3t1ont2()

        ## Get T2 new

        T2new = self.Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2) + self.T3onT2 + self.T3onT2sec + self.T4onT2

        T2new = np.einsum('uvpg,uvpg->uvpg', T2new, self.D,optimize=EINSUMOPT)


    
        ## Get T1 new
        
        T1new =    np.einsum('ui,ip->up',      giu, self.T1,                                   optimize=EINSUMOPT)
        T1new -=   np.einsum('ap,ua->up',      gap, self.T1,                                   optimize=EINSUMOPT)
        T1new -=   np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), self.T1, self.T1, optimize=EINSUMOPT)
        T1new -=   np.einsum('auip,ia->up',    2*(D2a - D2b) + D2c, self.T1,                   optimize=EINSUMOPT)
        T1new -=   np.einsum('aup->up',        F2a,                                            optimize=EINSUMOPT)
        T1new +=   np.einsum('uiip->up',       1.0/2.0*(E2a - E2b) + E2c,                      optimize=EINSUMOPT)
        T1new +=   np.einsum('uip->up',        C1,                                             optimize=EINSUMOPT)
        T1new -= 2*np.einsum('uipi->up',       D1,                                             optimize=EINSUMOPT)
        T1new += self.T3onT1
    
        T1new = np.einsum('up,up->up', T1new, self.d, optimize=EINSUMOPT)

        # DUMB FROZEN CORE
        if self.fcore:
            self.dumb_fc(T1new, T2new)
        
        ## Compute RESIDUES

        self.r1 = np.sum(np.sqrt(np.square(T1new - self.T1)))/(self.nvir*self.ndocc)

        self.r2 = np.sum(np.sqrt(np.square(T2new - self.T2)))/((self.nvir*self.ndocc)**2)
    
        ## Update amplitudes

        self.T1 = T1new
        self.T2 = T2new

    def relax_t3t1ont2(self):

        # This function will compute the contribution of disconnected T3*T1 terms on T2 equations. 
        # Since T1 are relaxed, this contribution changes and it will be updated by default during each iteration
        # This can add an extra cost to the CCSD iterations, but it can be turned off by setting RELAX_T3T1 = False

        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)
        AntiV = (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v]

        # First term
        X = 2*self.Vint[o,o,v,v] - self.Vint[o,o,v,v].swapaxes(2,3)
        X = np.einsum('mnef,me->nf', X, self.T1)
        X = np.einsum('nf, nijfab -> ijab', X, self.T3)
        self.T3onT2sec = copy.deepcopy(X)

        # Second term
        X = 0.5*np.einsum('mnef, njiebf -> ijmb', AntiV, self.T3) + np.einsum('mnef, jinfeb -> ijmb', self.Vint[o,o,v,v], self.T3)
        X = np.einsum('ijmb, ma -> ijab', X, self.T1)

        self.T3onT2sec += X

        # Third term
        X = 0.5*np.einsum('mnef, mjnfba -> ejab', AntiV, self.T3) + np.einsum('mnef, nmjbaf -> ejab', self.Vint[o,o,v,v], self.T3)
        X = np.einsum('ejab, ie -> ijab', X, self.T1)

        self.T3onT2sec += X
                
        # Apply permutation
        self.T3onT2sec += np.einsum('ijab -> jiba', self.T3onT2sec)

    def dumb_fc(self, T1, T2):
        f = slice(0,self.fcore) 
        T1[f] = 0.0
        T2[f,f] = 0.0

    def compute(self, CC_CONV=6, CC_MAXITER=50, E_CONV = 8, MP2_GUESS=False, RELAX_T3T1=True):

        # This is the main function of this code.
        # The correlation energy will be avaliable as example.Ecc
        # This function works in a 4 steps process:
        # - CASCI computation
        # - Translation step
        # - T3 and T4 contribution computation
        # - CCSD computation
        
        ############### STEP 1 ###############
        ##############  DETCI  ###############

        # Retrive CI coefficients and determinant objects from DETCI

        self.ref = Det(a = '1'*self.ndocc + '0'*self.nvir, b = '1'*self.ndocc + '0'*self.nvir)

        self.Ccas, self.determinants = read_ci_vec(self.ref, self.fdocc, self.nmo)

        ############### STEP 2 ###############
        ############  CASDecom  ##############

        # Run CASDecom to translate CI coefficients into CC amplitudes

        self.T1, self.T2, self.T3, self.T4abab, self.T4abaa = \
        CASDecom(self.Ccas, self.determinants, self.ref, fdocc = self.fdocc, fvir = self.fvir)

        # Compare CAS energy with the energy obtained from the translated amplitudes (They should be the same)

        self.cc_energy()
        #printout('CAS energy and initial TCC energy:')
        #printout('CAS Energy: {:<5.10f}'.format(self.Ecas))
        printout('CC Energy:  {:<5.10f}'.format(self.Ecc + self.Escf))

        ############### STEP 3 ###############
        ######  T3 & T4 contributions  #######

        # Compute contribution of high order amplitudes (T3 and T4) to T1 and T2 equations

        ## Slices: used to produce different sets of integrals within particles or holes space
        
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        ## Contribution of connected T3 terms to T1 equations. 
        ## Equation from Chem. Phys. Lett. 152, 382 (1988): G. E. Scuseria and H. F. Schaefer III

        V = (2*self.Vint - np.einsum('ijab -> ijba', self.Vint))[o,o,v,v]
        self.T3onT1 = np.einsum('ijab, jiupab -> up', V, self.T3)
        
        ## Contribution of connected T3 terms to T2 equations. 

        self.T3onT2 = +0.5*np.einsum('bmef, jimeaf -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[v,o,v,v], self.T3) \
                      +    np.einsum('bmef, ijmaef -> ijab', self.Vint[v,o,v,v], self.T3)                             \
                      -0.5*np.einsum('mnje, minbae -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,o,v], self.T3) \
                      -    np.einsum('mnje, imnabe -> ijab', self.Vint[o,o,o,v], self.T3)                             

        self.T3onT2 = self.T3onT2 + np.einsum('ijab -> jiba', self.T3onT2) 

        ## Contribution of disconnected T3*T1 terms to T2 equations via relax_t3t1ont2 function.

        self.relax_t3t1ont2()

        ## Contribution of connected T4 terms to T2 equations.
        
        self.T4onT2 = np.einsum('mnef, ijmnabef -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], self.T4abaa)        

        self.T4onT2 += np.einsum('ijab -> jiba', self.T4onT2)

        self.T4onT2 = (1.0/4.0)*self.T4onT2 + np.einsum('mnef, ijmnabef -> ijab', self.Vint[o,o,v,v], self.T4abab)

        ############### STEP 4 ###############
        ###############  CCSD  ###############

        # Build the Auxiliar Matrix D

        printout('Building Auxiliar D matrices...\n')
        t = time.time()
        self.D  = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])
        self.d  = np.zeros([self.ndocc, self.nvir])
        new = np.newaxis

        # Note that G. E. Scuseria et al. used a non conventional def. of D(i,a): ea - ei. Other refs will define this as (ei - ea)
        self.d = 1.0/(self.eps[new, v] - self.eps[o, new])
        self.D = 1.0/(self.eps[o, new, new, new] + self.eps[new, o, new, new] - self.eps[new, new, v, new] - self.eps[new, new, new, v])

        printout('Done. Time required: {:.5f} seconds'.format(time.time() - t))
        t = time.time()
        
        if MP2_GUESS:

            # If MP2_GUESS = True, MP2 initial T1 and T2 amplitudes will be used instead of the ones generated with CASCI

            self.T1 = np.zeros([self.ndocc, self.nvir])
            self.T2  = np.einsum('ijab,ijab->ijab', self.Vint[o,o,v,v], self.D)
            self.cc_energy()
            printout('MP2 Energy: {:<5.10f}'.format(self.Ecc+self.Escf))

        else:
            # Use MP2 guess for the amplitudes outisde CAS

            hold = copy.deepcopy(self.T2)
            self.T2  = np.einsum('ijab,ijab->ijab', self.Vint[o,o,v,v], self.D)
            h = slice(self.fdocc, self.ndocc)
            p = slice(0, self.nvir-self.fvir)
            self.T2[h,h,p,p] = hold[h,h,p,p]

        if self.read_amp:
            printout('\n Reading amplitudes from disk')
            self.T1 = read_T1(self.ndocc, self.nvir)
            self.T2 = read_T2(self.ndocc, self.nvir)
            printout('\n T1 read from disk:\n')
            printtensor(self.T1)
            printout('\n T2 read from disk:\n')
            printtensor(self.T2)
            self.cc_energy()
            printout('CC energy from disk amplitudes: {:<5.10f}'.format(self.Ecc+self.Escf))
            
        self.r1 = 1
        self.r2 = 1
        dE      = 1
        LIM = 10**(-CC_CONV)
        ELIM = 10**(-E_CONV)
        ite = 0

        if self.fcore:
            printout('\n Applying DUMBFC...')
            self.dumb_fc(self.T1, self.T2)

        printout('\n Starting CCSD Iterations')
        printout('='*36)

        tcc = time.time()
        while self.r2 > LIM or self.r1 > LIM or abs(dE) > ELIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            Eold = self.Ecc
            t = time.time()
            self.T1_T2_Update(RELAX_T3T1 = RELAX_T3T1)
            self.cc_energy()
            dE = self.Ecc - Eold
            printout("Iteration {}".format(ite))
            printout("Correlation energy:    {:<5.10f}".format(self.Ecc))
            printout("Energy change:         {:<5.10f}".format(dE))
            printout("T1 Residue:            {:>13.2E}".format(self.r1))
            printout("T2 Residue:            {:>13.2E}".format(self.r2))
            printout("Time required (s):     {:< 5.10f}".format(time.time() - t))
            printout('='*36)
        if self.write_amp:
            printout('\nWriting amplitudes...')
            write_T1(self.T1)
            write_T2(self.T2)
        tcc = time.time() - tcc 
        self.Ecc = self.Ecc + self.Escf
        printout("\nCC Equations Converged!!!")
        printout("Time required: {}".format(tcc))
        printout("Final CASCCSD Energy:     {:<5.10f}".format(self.Ecc))
