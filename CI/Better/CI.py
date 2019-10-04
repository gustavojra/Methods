import psi4
import numpy as np
import scipy.linalg as la
import os
import sys
import time
from itertools import permutations

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

file_dir = os.path.dirname('../')
sys.path.append(file_dir)

from fock import *
from Hamiltonian import *

def List2Str(lista):
    out = ''
    for i in lista:
        out += str(i)
    return out

class CI:
    
# Pull in Hartree-Fock data, including integrals

    def __init__(self, wfn):
        self.wfn = wfn
        self.nelec = wfn.nalpha() + wfn.nbeta()
        self.C = wfn.Ca()
        self.ndocc = wfn.doccpi()[0]
        self.nmo = wfn.nmo()
        self.nvir = self.nmo - self.ndocc
        self.eps = np.asarray(wfn.epsilon_a())
        self.nbf = self.C.shape[0]
        
        print("Number of Basis Functions:      {}".format(self.nbf))
        print("Number of Electrons:            {}".format(self.nelec))
        print("Number of Molecular Orbitals:   {}".format(self.nmo))
        print("Number of Doubly ocuppied MOs:  {}".format(self.ndocc))
    
        # Get Integrals
    
        print("Converting atomic integrals to MO integrals...")
        t = time.time()
        mints = psi4.core.MintsHelper(wfn.basisset())
        self.Vint = np.asarray(mints.mo_eri(self.C, self.C, self.C, self.C))
        self.h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.h = np.einsum('up,vq,uv->pq', self.C, self.C, self.h)
        # Convert to physicist notation
        #self.Vint = Vint.swapaxes(1,2)
        print("Completed in {} seconds!".format(time.time()-t))

        #self.orbitals = HF.orbitals
        #self.ndocc = HF.ndocc
        #self.nelec = HF.nelec
        #self.nbf = HF.nbf
        #self.virtual = self.nbf - self.ndocc
        #self.V_nuc = HF.V_nuc
        #self.h = HF.T + HF.V
        ## chemists
        #self.g = HF.g.swapaxes(1,2)
        #self.S = HF.S
        #print("Number of electrons: {}".format(self.nelec))
        #print("Number of basis functions: {}".format(self.nbf))
        #print("Number of doubly occupied orbitals: {}".format(self.ndocc))
        #print("Number of virtual spatial orbitals: {}".format(self.virtual))

    def compute_CIS(self):
        print("Starting CIS computation")
        oc = np.array([1]*self.ndocc + [0]*self.nvir)
        self.ref = Bra([oc, oc])
        self.determinants = [self.ref]
        
        # GENERATE EXCITATIONS
        
        print("Generating singly excited states")
        prog_total = self.nvir*self.ndocc*2
        prog = 0
        for i in range(self.ndocc, self.nbf):
            for a in range(self.ndocc):
                for s in [0, 1]:
                    new = self.ref.an(a, s).cr(i, s)
                    determinants.append(new)
                    prog += 1
                    print("Progress: {:2.0f}%".format(100*prog/prog_total))
        
        # COMPUTE HAMILTONIAN MATRIX
        
        print("Generating Hamiltonian Matrix")
        H = get_H(determinants, self.MIone, self.MItwo, v = True, t = True)
        
        # DIAGONALIZE HAMILTONIAN MATRIX
        
        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, C = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("Energies:")
        print(E)

    def compute_CISD(self):
        print("Starting CIS computation")
        oc = np.array([1]*self.ndocc + [0]*self.nvir)
        self.ref = Bra([oc, oc])
        determinants = [self.ref]

        # GENERATE EXCITATIONS

        print("Generating singly excited states")
        prog_total = self.nvir*self.ndocc*2
        prog = 0
        for i in range(self.ndocc, self.nbf):
            for a in range(self.ndocc):
                #for s in [0, 1]:
                for s in [0]:
                    new = self.ref.an(a, s).cr(i, s)
                    determinants.append(new)
                    prog += 1
            print("Progress: {:2.0f}%".format(100*prog/prog_total))

        singles_range = slice(1,len(determinants))

        print("Generating doubly excited states")
        prog_total = len(range(self.ndocc, self.nbf))
        prog = 0
        for i in range(self.ndocc, self.nbf):
            for j in range(self.ndocc, self.nbf):
                for a in range(self.ndocc):
                    for b in range(self.ndocc):
                        for s1 in [0, 1]:
                            for s2 in [0, 1]:
                                new = self.ref.an(a, s1).cr(i, s1).an(b, s2).cr(j,s2)
                                if new.p != 0 and new not in determinants:
                                    determinants.append(new)
            prog += 1
            print("Progress: {:2.0f}%".format(100*prog/prog_total))
        print("Number of Determinants: {}".format(len(determinants)))

        # COMPUTE HAMILTONIAN MATRIX

        print("Generating Hamiltonian Matrix")
        H = get_H(determinants, self.h, self.Vint, v = True, t = True)

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, C = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("Energies:")
        print("\nCISD Energy: {:<15.10f} ".format(E[0]) + emoji('viva'))
        x=C.T[singles_range]
        print(x)

    # CAS assuming nbeta = nalpha

    def compute_CAS(self, active_space ='',nfrozen=0, nvirtual=0):

        # Read active space
        space = []
        n_ac_orb = 0
        n_ac_elec_pair = int(self.nelec/2)
        print("Reading Active space")
        if active_space == 'full':
            active_space = 'a'*self.nmo
        if len(active_space) != self.nbf:
            raise NameError("Invalid active space. Please check the number of basis functions")
        for i in active_space:
            if i == 'o':
                space.append(int(1))
                n_ac_elec_pair -= 1
            elif i == 'a':
                space.append('a')
                n_ac_orb += 1
            elif i == 'u':
                space.append(int(0))
            else:
                raise NameError("Invalid active space entry: {}".format(i))

        active = np.array([1]*n_ac_elec_pair + [0]*(n_ac_orb - n_ac_elec_pair))
        print("Number of active orbitals: {}".format(n_ac_orb))
        print("Number of active electrons: {}".format(2*n_ac_elec_pair))

        # GENERATE DETERMINANTS

        print("Generating excitations")
        perms = set(permutations(active))
        print("Generating determinants")
        self.determinants = []
        for p1 in perms:
            for p2 in perms:
                alpha = space.copy()
                beta =  space.copy()
                for i,x in enumerate(np.where(np.array(space) == 'a')[0]):
                    alpha[x] = p1[i]
                    beta[x] = p2[i]
                self.determinants.append(Det(alpha=List2Str(alpha), beta=List2Str(beta)))
        print("Number of determinants: {}".format(len(self.determinants)))

        # COMPUTE HAMILTONIAN MATRIX

        print("Generating Hamiltonian Matrix")
        H = get_H(self.determinants, self.h, self.Vint, v = False, t = True)
        print(H[0])

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = timeit.default_timer()
        E, Ccas = la.eigh(H)
        tf = timeit.default_timer()
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Electronic Energy:")
        self.E = E[0]
        self.C0 = Ccas[:,0]
        self.Ecas = E[0]
        print(self.Ecas)
        return self.Ecas

if __name__ == '__main__':
        
    # Input Geometry    
    
    H2 = psi4.geometry("""
        0 1
        H 
        H 1 0.76
        symmetry c1
    """)

    #He2 = psi4.geometry("""
    #    0 1
    #    He 
    #    He 1 1.0
    #    symmetry c1
    #""")
    
    #water = psi4.geometry("""
    #    0 1
    #    O
    #    H 1 0.96
    #    H 1 0.96 2 104.5
    #    symmetry c1
    #""")
    
    #ethane = psi4.geometry("""
    #    0 1
    #    C       -3.4240009952      1.7825072183      0.0000001072                 
    #    C       -1.9048206760      1.7825072100     -0.0000000703                 
    #    H       -3.8005812586      0.9031676785      0.5638263076                 
    #    H       -3.8005814434      1.7338892156     -1.0434433083                 
    #    H       -3.8005812617      2.7104647651      0.4796174543                 
    #    H       -1.5282404125      0.8545496587     -0.4796174110                 
    #    H       -1.5282402277      1.8311252186      1.0434433449                 
    #    H       -1.5282404094      2.6618467448     -0.5638262767  
    #    symmetry c1
    #""")
    
    #form = psi4.geometry("""
    #0 1
    #O
    #C 1 1.22
    #H 2 1.08 1 120.0
    #H 2 1.08 1 120.0 3 -180.0
    #symmetry c1
    #""")
    
    # Basis set
    
    basis = '3-21g'
    
    # Psi4 Options
    
    psi4.core.be_quiet()
    psi4.set_options({'basis': basis,
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'e_convergence' : 1e-10,
                      'freeze_core': 'false'})
    
    e_scf, wfn = psi4.energy('SCF', return_wfn=True)
    
    t = time.time()
    CAS = CI(wfn)
    Enuc = H2.nuclear_repulsion_energy()
    
    #CAS.compute_CISD()
    Ecas = CAS.compute_CAS('full')
    print("\nCAS Energy: {:<15.10f} ".format(Ecas+Enuc) + emoji('whale'))
    print("Time required: {} seconds.".format(time.time()-t))
