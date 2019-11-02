import psi4
import os
import sys
import numpy as np
import scipy.linalg as la
import time
import copy
from itertools import permutations

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *
from fock import *
from Hamiltonian import *
from ampcomp import tcompare

np.set_printoptions(suppress=True)

class HTCCSD:

    def __init__(self, mol):

        # Run SCF information 

        self.Escf, wfn = psi4.energy('scf', return_wfn = True)
        self.nelec = wfn.nalpha() + wfn.nbeta()
        self.C = wfn.Ca()
        self.ndocc = wfn.doccpi()[0]
        self.nmo = wfn.nmo()
        self.nvir = self.nmo - self.ndocc
        self.eps = np.asarray(wfn.epsilon_a())
        self.nbf = self.C.shape[0]
        self.Vnuc = mol.nuclear_repulsion_energy()
        
        print("Number of Electrons:            {}".format(self.nelec))
        print("Number of Basis Functions:      {}".format(self.nbf))
        print("Number of Molecular Orbitals:   {}".format(self.nmo))
        print("Number of Doubly ocuppied MOs:  {}\n".format(self.ndocc))
    
        # Get Integrals.
    
        print("Converting atomic integrals to MO integrals...")
        t = time.time()
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.Vint = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
        self.Vint = self.Vint.swapaxes(1,2) # Convert to Physicists' notation
        self.h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.h = np.einsum('up,vq,uv->pq', self.C, self.C, self.h)
        print("Completed in {} seconds!".format(time.time()-t))

    
    def CAS(self, active_space='', show_prog = False):

        # Read active space and determine number of active electrons. 
        # Format: sequence of letters ordered
        # according to orbital energies. 
        # Legend:
        # o = frozen doubly occupied orbital
        # a = active orbital
        # u = frozen unnocupied orbital

        template_space = ''
        n_ac_orb = 0
        n_ac_elec_pair = int(self.nelec/2)
        print("Reading Active space")
        if active_space == 'full':
            active_space = 'a'*self.nmo
        if len(active_space) != self.nbf:
            raise NameError("Invalid active space format. Please check the number of basis functions.")
        for i in active_space:
            if i == 'o':
                template_space += '1'
                n_ac_elec_pair -= 1
            elif i == 'a':
                template_space += '{}'
                n_ac_orb += 1
            elif i == 'u':
                template_space += '0'
            else:
                raise NameError("Invalid active space entry: {}".format(i))

        if n_ac_elec_pair < 0:
                raise NameError("Negative number of active electrons")

        # Produce a reference determinant

        self.ref = Det(a = ('1'*self.ndocc + '0'*self.nvir), \
                       b = ('1'*self.ndocc + '0'*self.nvir))

        print("Number of active orbitals: {}".format(n_ac_orb))
        print("Number of active electrons: {}\n".format(2*n_ac_elec_pair))

        # Use permutations to generate strings that will represent excited determinants

        print("Generating excitations...")
        perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))
        print("Done.\n")

        # Use the strings to generate Det objects. Use second quantization to attribute signs 

        self.determinants = []
        progress = 0
        file = sys.stdout
        for p1 in perms:
            for p2 in perms:
                self.determinants.append(Det(a=template_space.format(*p1), \
                                             b=template_space.format(*p2), \
                                                    ref=self.ref,sq=True))
            progress += 1
            showout(progress, len(perms), 50, "Generating Determinants: ", file)
        file.write('\n')
        file.flush()
        print("Number of determinants: {}".format(len(self.determinants)))

        # Construct the Hamiltonian Matrix
        # Note: Input for two electron integral must be using Chemists' notation

        H = get_H(self.determinants, self.h, self.Vint.swapaxes(1,2), v = True, t = True)

        # Diagonalize the Hamiltonian Matrix

        print("Diagonalizing Hamiltonian Matrix")
        t0 = time.time()
        E, Ccas = la.eigh(H)
        tf = time.time()
        self.Ecas = E[0] + self.Vnuc
        self.Ccas = Ccas[:,0]
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Energy: {:<5.10f}".format(self.Ecas))

    def cc_energy(self):
    
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)
        tau = self.T2 + np.einsum('ia,jb->ijab', self.T1, self.T1)
        X = 2*tau - np.einsum('ijab->jiab',tau)
        self.Ecc = np.einsum('abij,ijab->', self.Vint[v,v,o,o], X)

    def T1_T2_Update(self, RELAX_T3 = True, EINSUMOPT='optimal'):
    
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
        
        ### UPDATE T3 EFFECT
        if RELAX_T3:
            X = + np.einsum('me, njifba -> ijmnabef', self.T1, self.CAS_T3aba) \
                - np.einsum('ma, ijnebf -> ijmnabef', self.T1, self.CAS_T3aba) \
                - np.einsum('ie, mjnabf -> ijmnabef', self.T1, self.CAS_T3aba)

            X = np.einsum('mnef, ijmnabef -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], X)

            Y = + np.einsum('me, nijfab -> ijmnabef', self.T1, self.CAS_T3aba) \
                - np.einsum('ma, nijfeb -> ijmnabef', self.T1, self.CAS_T3aba) \
                - np.einsum('ie, nmjfab -> ijmnabef', self.T1, self.CAS_T3aba)

            Y = np.einsum('mnef, ijmnabef -> ijab', self.Vint[o,o,v,v], Y)

            Z = + np.einsum('mb, ijnefa -> ijmnabef', self.T1, self.CAS_T3aba) \
                - np.einsum('ie, mnjfab -> ijmnabef', self.T1, self.CAS_T3aba) \

            Z = np.einsum('mnfe, ijmnabef -> ijab', self.Vint[o,o,v,v], Z)

            self.T3onT2sec = X + Y + Z + np.einsum('ijab -> jiba', X + Y + Z)
        ### END
    
        T2new = self.Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2) + self.T3onT2 + self.T3onT2sec + self.T4onT2 
    
        T2new = np.einsum('uvpg,uvpg->uvpg', T2new, self.D,optimize=EINSUMOPT)

        self.r2 = np.sum(np.abs(T2new - self.T2))
    
        # T1 Amplitudes update
        
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
        
        self.r1 = np.sum(np.abs(T1new - self.T1)) 
    
        self.T1, self.T2 = T1new, T2new

    # PRINT FUNCTIONS FOR DEBUGGING

    def printcast1(self,w=False):
        out = ''
        for i in range(self.ndocc):
            for a in range(self.nvir):
                x = self.CAS_T1[i,a]
                if abs(x) > 1e-10:
                    out += '{}a -> {}a {:< 5.8e}'.format(i+1,1+a+self.ndocc,x)
                    out += '\n'
        if w:
            fil = open('t1amps.dat','w') 
            fil.write(out)
            fil.close()
        else:
            print(out)

    def printcast2(self,w=False):
        out = ''
        for i in range(self.ndocc):
            for j in range(self.ndocc):
                if j > i: break
                for a in range(self.nvir):
                    for b in range(self.nvir):
                        if b > a: break
                        x = self.CAS_T2[i,j,a,b]
                        if abs(x) > 1e-10:
                            out += '{}a {}b -> {}a {}b {:< 5.8e}'.format(i+1,j+1,1+a+self.ndocc,1+b+self.ndocc,x)
                            out += '\n'
        if w:
            fil = open('t2amps.dat','w') 
            fil.write(out)
            fil.close()
        else:
            print(out)

    def printcast3(self,w=False):
        out = ''
        for i in range(self.ndocc):
            for j in range(self.ndocc):
                if j > i: break
                for k in range(self.ndocc):
                    if k > j: break
                    for a in range(self.nvir):
                        for b in range(self.nvir):
                            if b > a: break
                            for c in range(self.nvir):
                                if c > b: break
                                x = self.CAS_T3aba[i,j,k,a,b,c]
                                if abs(x) > 1e-10:
                                    out += '{}a {}b {}a -> {}a {}b {}a  {:< 5.8e}'.format(i+1,j+1,k+1,1+a+self.ndocc,1+b+self.ndocc,1+c+self.ndocc,x)
                                    out += '\n'
                                y = self.CAS_T3aaa[i,j,k,a,b,c]
                                if abs(y) > 1e-10:
                                    out += '{}a {}a {}a -> {}a {}a {}a  {:< 5.8e}'.format(i+1,j+1,k+1,1+a+self.ndocc,1+b+self.ndocc,1+c+self.ndocc,y)
                                    out += '\n'
        if w:
            fil = open('t3amps.dat','w') 
            fil.write(out)
            fil.close()
        else:
            print(out)

    def printcast4(self,w=False):
        out = ''
        for i in range(self.ndocc):
            for j in range(self.ndocc):
                if j > i: break
                for k in range(self.ndocc):
                    if k > j: break
                    for a in range(self.nvir):
                        for b in range(self.nvir):
                            if b > a: break
                            for c in range(self.nvir):
                                if c > b: break
                                for d in range(self.nvir):
                                    if d > c: break
                                    for l in range(self.ndocc):
                                        if l > k: break
                                        x = self.CAS_T4abaa[i,j,k,l,a,b,c,d]
                                        if abs(x) > 1e-10:
                                            out += '{}a {}b {}a {}a -> {}a {}b {}a {}a {:< 5.8e}'.format(i+1,j+1,k+1,l+1,1+a+self.ndocc,1+b+self.ndocc,1+c+self.ndocc,1+d+self.ndocc,x)
                                            out += '\n'
                                        x = self.CAS_T4abab[i,j,k,l,a,b,c,d]
                                        if abs(x) > 1e-6:
                                            out += '{}a {}b {}a {}b -> {}a {}b {}a {}b {:< 5.8e}'.format(i+1,j+1,k+1,l+1,1+a+self.ndocc,1+b+self.ndocc,1+c+self.ndocc,1+d+self.ndocc,x)
                                            out += '\n'
        if w:
            fil = open('t4amps.dat','w') 
            fil.write(out)
            fil.close()
        else:
            print(out)

    def HTCCSD(self, active_space='', CC_CONV=6, CC_MAXITER=50, MP2_GUESS=False, RELAX_T3=True):
        
        # Compute CAS

        # If active_space is a list it will be interpred as indexes for active orbitals        
        tinit = time.time()

        if type(active_space) == list:
            indexes = copy.deepcopy(active_space)
            active_space = ''
            hf_string = 'o'*self.ndocc + 'u'*self.nvir
            for i in range(self.nbf):
                if i in indexes:
                    active_space += 'a'
                else:
                    active_space += hf_string[i]

        # none is none

        if active_space == 'none':
            active_space = 'o'*self.ndocc + 'u'*self.nvir

        # Setting active_space = 'full' calls Full CI

        if active_space == 'full':
            active_space = 'a'*self.nmo

        print('------- COMPLETE ACTIVE SPACE CONFIGURATION INTERACTION STARTED -------\n')

        self.CAS(active_space)
        
        print('------- COMPLETE ACTIVE SPACE CONFIGURATION INTERACTION FINISHED -------\n')

        print('Collecting C1 and C2 coefficients...\n')

        C0 = self.Ccas[self.determinants.index(self.ref)]

        if abs(C0) < 0.01:
            raise NameError('Leading Coefficient too small C0 = {}\n Restricted orbitals not appropriate.'.format(C0))
        # Determine indexes for the CAS space

        self.Ccas = self.Ccas/C0

        self.CAS_holes = []
        for i,x in enumerate(active_space[0:self.ndocc]):
            if x == 'a':
                self.CAS_holes.append(i)

        self.CAS_particles = []
        for i,x in enumerate(active_space[self.ndocc:]):
            if x == 'a':
                self.CAS_particles.append(i)

        self.CAS_T1 = np.zeros([self.ndocc, self.nvir])
        self.CAS_T2 = np.zeros([self.ndocc, self.ndocc, self.nvir, self.nvir])

        self.CAS_T3aaa = np.zeros([self.ndocc, self.ndocc, self.ndocc, self.nvir, self.nvir, self.nvir])
        self.CAS_T3aba = np.zeros([self.ndocc, self.ndocc, self.ndocc, self.nvir, self.nvir, self.nvir])

        self.CAS_T4abaa = np.zeros([self.ndocc, self.ndocc, self.ndocc, self.ndocc, self.nvir, self.nvir, self.nvir, self.nvir])
        self.CAS_T4abab = np.zeros([self.ndocc, self.ndocc, self.ndocc, self.ndocc, self.nvir, self.nvir, self.nvir, self.nvir])

        # Search for the appropriate coefficients using a model Determinant

        # Singles: Only the case (alpha -> alpha) is necessary

        for i in self.CAS_holes:
            for a in self.CAS_particles:
                search = self.ref.copy()
            
                # anh -> i
                search.anh(i, spin=0)

                # cre -> a
                search.cre(a+self.ndocc, spin=0)

                index = self.determinants.index(search)
                CASdet = self.determinants[index]
                # The phase guarentees that the determinant in the CI framework is the same as the one created in the CC framework
                self.CAS_T1[i,a] = self.Ccas[index] * search.sign() * CASdet.sign()

        # Doubles: Only the case (alpha, beta -> alpha, beta) is necessary

        for i in self.CAS_holes:
            for a in self.CAS_particles:
                for j in self.CAS_holes:
                    for b in self.CAS_particles:
                        search = self.ref.copy()

                        # anh -> i
                        search.anh(i, spin=0)

                        # anh -> j
                        search.anh(j, spin=1)

                        # cre -> b
                        search.anh(b + self.ndocc, spin=1)

                        # cre -> a
                        search.cre(a + self.ndocc, spin=0)

                        index = self.determinants.index(search)
                        CASdet = self.determinants[index]
                        # The phase guarentees that the determinant in the CI framework is the same as the one created in the CC framework
                        self.CAS_T2[i,j,a,b] = self.Ccas[index] * search.sign() * CASdet.sign()

        # Triples. Two cases are going to be considered: aaa -> aaa, abb -> abb

        # First Case: aaa -> aaa

        for i in self.CAS_holes:
            for a in self.CAS_particles:
                for j in self.CAS_holes:
                    for b in self.CAS_particles:
                        for k in self.CAS_holes:
                            for c in self.CAS_particles:
                                # Since all index are alphas they cannot be the same
                                if i == j or i == k or j == k:
                                    continue
                                if a == b or b == c or a == c:
                                    continue
                                search = self.ref.copy()

                                # anh -> i (alpha)
                                search.anh(i, spin=0)

                                # anh -> j (alpha)
                                search.anh(j, spin=0)

                                # anh -> k (alpha)
                                search.anh(k, spin=0)

                                # cre -> c (alpha)
                                search.cre(c + self.ndocc, spin=0)

                                # cre -> b (alpha)
                                search.cre(b + self.ndocc, spin=0)

                                # cre -> a (alpha)
                                search.cre(a + self.ndocc, spin=0)

                                index = self.determinants.index(search)
                                CASdet = self.determinants[index]
                                self.CAS_T3aaa[i,j,k,a,b,c] = self.Ccas[index] * search.sign() * CASdet.sign()

        # Second Case: aba -> aba

        for i in self.CAS_holes:
            for a in self.CAS_particles:
                for j in self.CAS_holes:
                    for b in self.CAS_particles:
                        for k in self.CAS_holes:
                            for c in self.CAS_particles:
                                # Since ik and ac are alphas, they cannot be the same
                                if i == k or a == c:
                                    continue
                                search = self.ref.copy()

                                # anh -> i (alpha)
                                search.anh(i, spin=0)

                                # anh -> j (beta)
                                search.anh(j, spin=1)

                                # anh -> k (alpha)
                                search.anh(k, spin=0)

                                # cre -> c (alpha)
                                search.cre(c + self.ndocc, spin=0)

                                # cre -> b (beta)
                                search.cre(b + self.ndocc, spin=1)

                                # cre -> a (alpha)
                                search.cre(a + self.ndocc, spin=0)

                                index = self.determinants.index(search)
                                CASdet = self.determinants[index]
                                self.CAS_T3aba[i,j,k,a,b,c] = self.Ccas[index] * search.sign() * CASdet.sign()

        # Quadruples 

        ## First case: abaa -> abaa

        for i in self.CAS_holes:
            for a in self.CAS_particles:
                for j in self.CAS_holes:
                    for b in self.CAS_particles:
                        for k in self.CAS_holes:
                            for c in self.CAS_particles:
                                for l in self.CAS_holes:
                                    for d in self.CAS_particles:
                                        # Same spin indexes cannot be the same
                                        if i == k or i == l or k == l:
                                            continue
                                        if a == c or a == d or c == d:
                                            continue
                                        search = self.ref.copy()

                                        # anh -> i (alpha)
                                        search.anh(i, spin=0)

                                        # anh -> j (beta)
                                        search.anh(j, spin=1)

                                        # anh -> k (alpha)
                                        search.anh(k, spin=0)

                                        # anh -> l (alpha)
                                        search.anh(l, spin=0)

                                        # cre -> d (alpha)
                                        search.cre(d + self.ndocc, spin=0)

                                        # cre -> c (alpha)
                                        search.cre(c + self.ndocc, spin=0)

                                        # cre -> b (beta)
                                        search.cre(b + self.ndocc, spin=1)

                                        # cre -> a (alpha)
                                        search.cre(a + self.ndocc, spin=0)

                                        index = self.determinants.index(search)
                                        CASdet = self.determinants[index]
                                        self.CAS_T4abaa[i,j,k,l,a,b,c,d] = self.Ccas[index] * search.sign() * CASdet.sign()

        ## Second case: abab -> abab

        for i in self.CAS_holes:
            for a in self.CAS_particles:
                for j in self.CAS_holes:
                    for b in self.CAS_particles:
                        for k in self.CAS_holes:
                            for c in self.CAS_particles:
                                for l in self.CAS_holes:
                                    for d in self.CAS_particles:
                                        # Same spin indexes cannot be the same
                                        if i == k or j == l:
                                            continue
                                        if a == c or b == d:
                                            continue
                                        search = self.ref.copy()

                                        # anh -> i (alpha)
                                        search.anh(i, spin=0)

                                        # anh -> j (beta)
                                        search.anh(j, spin=1)

                                        # anh -> k (alpha)
                                        search.anh(k, spin=0)

                                        # anh -> l (beta)
                                        search.anh(l, spin=1)

                                        # cre -> d (beta)
                                        search.cre(d + self.ndocc, spin=1)

                                        # cre -> c (alpha)
                                        search.cre(c + self.ndocc, spin=0)

                                        # cre -> b (beta)
                                        search.cre(b + self.ndocc, spin=1)

                                        # cre -> a (alpha)
                                        search.cre(a + self.ndocc, spin=0)

                                        index = self.determinants.index(search)
                                        CASdet = self.determinants[index]
                                        self.CAS_T4abab[i,j,k,l,a,b,c,d] = self.Ccas[index] * search.sign() * CASdet.sign()

        # Translate CI coefficients into CC amplitudes

        print('Translating CI coefficients into CC amplitudes...\n')

        # Singles: already done

        # Doubles
        self.CAS_T2 = self.CAS_T2 - np.einsum('ia,jb-> ijab', self.CAS_T1, self.CAS_T1)

        # Triples
        ## First case aaa -> aaa
        self.CAS_T3aaa = self.CAS_T3aaa

        T2aa = self.CAS_T2 - np.einsum('ijab -> jiab', self.CAS_T2)
        print('Test t2aa')
        print(T2aa[0,1,0,1])
        print(T2aa[1,0,0,1])
        print(T2aa[0,1,1,0])
        print(T2aa[1,0,1,0])

        self.CAS_T3aaa += - np.einsum('ia,jkbc ->ijkabc',self.CAS_T1,T2aa)  + np.einsum('ib,jkac -> ijkabc', self.CAS_T1,T2aa) - np.einsum('ic,jkab -> ijkabc', self.CAS_T1,T2aa) \
                          + np.einsum('ja,ikbc ->ijkabc',self.CAS_T1,T2aa)  - np.einsum('jb,ikac -> ijkabc', self.CAS_T1,T2aa) + np.einsum('jc,ikab -> ijkabc', self.CAS_T1,T2aa) \
                          - np.einsum('ka,ijbc ->ijkabc',self.CAS_T1,T2aa)  + np.einsum('kb,ijac -> ijkabc', self.CAS_T1,T2aa) - np.einsum('kc,ijab -> ijkabc', self.CAS_T1,T2aa) 

        self.CAS_T3aaa += - np.einsum('ia,jb,kc -> ijkabc', self.CAS_T1,self.CAS_T1,self.CAS_T1) + np.einsum('ia,jc,kb -> ijkabc', self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                          + np.einsum('ib,ja,kc -> ijkabc', self.CAS_T1,self.CAS_T1,self.CAS_T1) - np.einsum('ib,jc,ka -> ijkabc', self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                          - np.einsum('ic,ja,kb -> ijkabc', self.CAS_T1,self.CAS_T1,self.CAS_T1) + np.einsum('ic,jb,ka -> ijkabc', self.CAS_T1,self.CAS_T1,self.CAS_T1) 

        ## Second case aba -> aba

        self.CAS_T3aba += - np.einsum('ia,jkbc -> ijkabc', self.CAS_T1, self.CAS_T2) \
                          + np.einsum('ic,kjab -> ijkabc', self.CAS_T1, self.CAS_T2) \
                          + np.einsum('ka,ijcb -> ijkabc', self.CAS_T1, self.CAS_T2) \
                          - np.einsum('kc,ijab -> ijkabc', self.CAS_T1, self.CAS_T2) \
                          - np.einsum('jb,ikac -> ijkabc', self.CAS_T1, T2aa)

        self.CAS_T3aba += - np.einsum('ia,jb,kc -> ijkabc', self.CAS_T1, self.CAS_T1, self.CAS_T1) \
                          + np.einsum('ic,jb,ka -> ijkabc', self.CAS_T1, self.CAS_T1, self.CAS_T1) 

        # Quadruples

        ## First case abaa -> abaa

        self.CAS_T4abaa = self.CAS_T4abaa

        ### T1 * T3 terms

        self.CAS_T4abaa += - np.einsum('ia, kjlcbd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('ic, kjlabd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('id, kjlabc -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('jb, iklacd -> ijklabcd', self.CAS_T1, self.CAS_T3aaa) \
                           + np.einsum('ka, ijlcbd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('kc, ijlabd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('kd, ijlabc -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('la, ijkcbd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('lc, ijkabd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('ld, ijkabc -> ijklabcd', self.CAS_T1, self.CAS_T3aba)

        ### T2 * T2 terms

        self.CAS_T4abaa += - np.einsum('ijab, klcd -> ijklabcd', self.CAS_T2, T2aa) \
                           + np.einsum('ijcb, klad -> ijklabcd', self.CAS_T2, T2aa) \
                           - np.einsum('ijdb, klac -> ijklabcd', self.CAS_T2, T2aa) \
                           - np.einsum('ljdb, ikac -> ijklabcd', self.CAS_T2, T2aa) \
                           + np.einsum('ljcb, ikad -> ijklabcd', self.CAS_T2, T2aa) \
                           - np.einsum('ljab, ikcd -> ijklabcd', self.CAS_T2, T2aa) \
                           + np.einsum('kjdb, ilac -> ijklabcd', self.CAS_T2, T2aa) \
                           - np.einsum('kjcb, ilad -> ijklabcd', self.CAS_T2, T2aa) \
                           + np.einsum('kjab, ilcd -> ijklabcd', self.CAS_T2, T2aa) 
        
        ### T1 * T1 * T2 terms

        self.CAS_T4abaa += - np.einsum('jb,ia,klcd -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           + np.einsum('jb,ic,klad -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           - np.einsum('jb,id,klac -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           + np.einsum('jb,ka,ilcd -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           - np.einsum('jb,kc,ilad -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           + np.einsum('jb,kd,ilac -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           - np.einsum('jb,la,ikcd -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           + np.einsum('jb,lc,ikad -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           - np.einsum('jb,ld,ikac -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa) \
                           - np.einsum('ia,kc,ljdb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ia,kd,ljcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ia,lc,kjdb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ia,ld,kjcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ic,ka,ljdb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ic,kd,ljab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ic,la,kjdb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ic,ld,kjab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('id,ka,ljcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('id,kc,ljab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('id,la,kjcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('id,lc,kjab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ka,lc,ijdb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ka,ld,ijcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('kc,la,ijdb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('kc,ld,ijab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('kd,la,ijcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('kd,lc,ijab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2)

        ### T1 * T1 * T1 * T1 terms

        self.CAS_T4abaa += - np.einsum('jb, ia, kc, ld -> ijkabcd', self.CAS_T1,self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                           + np.einsum('jb, ia, kd, lc -> ijkabcd', self.CAS_T1,self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                           + np.einsum('jb, ic, ka, ld -> ijkabcd', self.CAS_T1,self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                           - np.einsum('jb, ic, kd, la -> ijkabcd', self.CAS_T1,self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                           - np.einsum('jb, id, ka, lc -> ijkabcd', self.CAS_T1,self.CAS_T1,self.CAS_T1,self.CAS_T1) \
                           + np.einsum('jb, id, kc, la -> ijkabcd', self.CAS_T1,self.CAS_T1,self.CAS_T1,self.CAS_T1) 
        
        ## Second case: abab -> abab

        self.CAS_T4_abab = self.CAS_T4abab

        ### T1 * T3 terms

        self.CAS_T4abab += - np.einsum('ia, jklbcd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('ic, jklbad -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('jb, ilkadc -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('jd, ilkabc -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('ka, jilbcd -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('kc, jilbad -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           + np.einsum('lb, ijkadc -> ijklabcd', self.CAS_T1, self.CAS_T3aba) \
                           - np.einsum('ld, ijkabc -> ijklabcd', self.CAS_T1, self.CAS_T3aba)

        ### T2 * T2 terms

        self.CAS_T4abab += - np.einsum('ijab, klcd -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           + np.einsum('ijad, klcb -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           + np.einsum('ijcb, klad -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           - np.einsum('ijcd, klab -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           - np.einsum('ikac, jlbd -> ijklabcd', T2aa, T2aa)               \
                           + np.einsum('ilab, kjcd -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           - np.einsum('ilad, jkbc -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           - np.einsum('ilcb, kjad -> ijklabcd', self.CAS_T2, self.CAS_T2) \
                           + np.einsum('ilcd, kjab -> ijklabcd', self.CAS_T2, self.CAS_T2) 

        ### T1 * T1 * T2 terms

        self.CAS_T4abab += - np.einsum('ia,jb,klcd -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ia,jd,klcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ia,kc,jlbd -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa)        \
                           + np.einsum('ia,lb,kjcd -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ia,ld,jkbc -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ic,jb,klad -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('ic,jd,klab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ic,ka,jlbd -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa)        \
                           - np.einsum('ic,lb,kjad -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ic,ld,kjab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('jb,ka,ilcd -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('jb,kc,ilad -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('jb,ld,ikac -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa)        \
                           - np.einsum('jd,ka,ilcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('jd,kc,ilab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('jd,lb,ikac -> ijklabcd', self.CAS_T1, self.CAS_T1, T2aa)        \
                           - np.einsum('ka,lb,ijcd -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('ka,ld,ijcb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           + np.einsum('kc,lb,ijad -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) \
                           - np.einsum('kc,ld,ijab -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T2) 

        ### T1 * T1 * T1 * T1 terms

        self.CAS_T4abab += - np.einsum('ia,jb,kc,ld -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T1, self.CAS_T1) \
                           + np.einsum('ia,jd,kc,lb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T1, self.CAS_T1) \
                           + np.einsum('ic,jb,ka,ld -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T1, self.CAS_T1) \
                           - np.einsum('ic,jd,ka,lb -> ijklabcd', self.CAS_T1, self.CAS_T1, self.CAS_T1, self.CAS_T1)

        self.printcast4()

        self.T2 = self.CAS_T2
        self.T1 = self.CAS_T1
        self.cc_energy()
        print('\n')
        print('CAS Energy: {:<5.10f}'.format(self.Ecas))
        print('TCC Energy: {:<5.10f}'.format(self.Ecc + self.Escf))

        # Slices
        
        o = slice(0, self.ndocc)
        v = slice(self.ndocc, self.nbf)

        # Compute T3 contribution to T1. Checked!! PRETTY SURE THIS IS CORRECT!

        self.T3onT1 = + (1.0/4.0)*np.einsum('mnef,imnaef->ia', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], self.CAS_T3aaa) \
                      + (1.0/4.0)*np.einsum('mnef,mineaf->ia', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], self.CAS_T3aba) \
                      + np.einsum('mnef,imnaef->ia', self.Vint[o,o,v,v], self.CAS_T3aba) 

        # Compute T3 contribution to T2. Checked!! Spin-Integrated equation simplified with permutation operator.

        self.T3onT2 = +    np.einsum('bmef, ijmaef -> ijab', self.Vint[v,o,v,v], self.CAS_T3aba)                             \
                      +0.5*np.einsum('bmef, jimeaf -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[v,o,v,v], self.CAS_T3aba) \
                      -    np.einsum('mnje, imnabe -> ijab', self.Vint[o,o,o,v], self.CAS_T3aba)                             \
                      -0.5*np.einsum('mnje, minbae -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,o,v], self.CAS_T3aba)

        self.T3onT2 = self.T3onT2 + np.einsum('ijab -> jiba', self.T3onT2) 

        ## T1 dependend part

        X = + np.einsum('me, njifba -> ijmnabef', self.T1, self.CAS_T3aba) \
            - np.einsum('ma, ijnebf -> ijmnabef', self.T1, self.CAS_T3aba) \
            - np.einsum('ie, mjnabf -> ijmnabef', self.T1, self.CAS_T3aba)

        X = np.einsum('mnef, ijmnabef -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], X)

        Y = + np.einsum('me, nijfab -> ijmnabef', self.T1, self.CAS_T3aba) \
            - np.einsum('ma, nijfeb -> ijmnabef', self.T1, self.CAS_T3aba) \
            - np.einsum('ie, nmjfab -> ijmnabef', self.T1, self.CAS_T3aba)

        Y = np.einsum('mnef, ijmnabef -> ijab', self.Vint[o,o,v,v], Y)

        Z = + np.einsum('mb, ijnefa -> ijmnabef', self.T1, self.CAS_T3aba) \
            - np.einsum('ie, mnjfab -> ijmnabef', self.T1, self.CAS_T3aba) \

        Z = np.einsum('mnfe, ijmnabef -> ijab', self.Vint[o,o,v,v], Z)

        self.T3onT2sec = X + Y + Z + np.einsum('ijab -> jiba', X + Y + Z)

        # Compute T4 contribution to T2
        
        self.T4onT2 = + np.einsum('mnef, ijmnabef -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], self.CAS_T4abaa) \
                      + np.einsum('mnef, jimnbaef -> ijab', (self.Vint - self.Vint.swapaxes(2,3))[o,o,v,v], self.CAS_T4abaa) \
                      + np.einsum('mnef, ijmnabef -> ijab', self.Vint[o,o,v,v], self.CAS_T4abab) \
                      + np.einsum('mnef, ijnmabfe -> ijab', self.Vint[o,o,v,v], self.CAS_T4abab) \
                      + np.einsum('mnfe, ijnmabef -> ijab', self.Vint[o,o,v,v], self.CAS_T4abab) \
                      + np.einsum('mnfe, ijmnabfe -> ijab', self.Vint[o,o,v,v], self.CAS_T4abab)    # these terms can prob be condensed into one
        
        self.T4onT2 *= (1.0/4.0)
    
        # Compute CCSD 

        print('------- TAILORED COUPLED CLUSTER STARTED -------\n')


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
        
        self.cc_energy()
        print('CC Energy from CAS Amplitudes: {:<5.10f}'.format(self.Ecc+self.Escf))

        tcompare(self.T1, self.T2, self.CAS_T3aaa, self.CAS_T3aba, self.CAS_T4abaa, self.CAS_T4abab, self.ndocc)

        # Guess MP2

        if MP2_GUESS:

            self.T1 = np.zeros([self.ndocc, self.nvir])
            self.T2  = np.einsum('ijab,ijab->ijab', self.Vint[o,o,v,v], self.D)
            self.cc_energy()
            print('MP2 Energy: {:<5.10f}'.format(self.Ecc+self.Escf))
            
        self.r1 = 1
        self.r2 = 1
            
        LIM = 10**(-CC_CONV)
        ite = 0
        
        self.printcast1(w=True)
        self.printcast2(w=True)
        self.printcast3(w=True)
        self.printcast4(w=True)
        
        while self.r2 > LIM or self.r1 > LIM:
            ite += 1
            if ite > CC_MAXITER:
                raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
            Eold = self.Ecc
            t = time.time()
            self.T1_T2_Update(RELAX_T3 = RELAX_T3)
            self.cc_energy()
            dE = self.Ecc - Eold
            print('-'*50)
            print("Iteration {}".format(ite))
            print("CC Correlation energy: {}".format(self.Ecc))
            print("Energy change:         {}".format(dE))
            print("T1 Residue:            {}".format(self.r1))
            print("T2 Residue:            {}".format(self.r2))
            print("Time required:         {}".format(time.time() - t))
            print('-'*50)
        
        self.Ecc = self.Ecc + self.Escf

        print("\nTCC Equations Converged!!!")
        print("Final TCCSD Energy:     {:<5.10f}".format(self.Ecc))
        print("Total Computation time:        {}".format(time.time() - tinit))
