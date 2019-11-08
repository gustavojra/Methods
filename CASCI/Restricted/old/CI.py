import psi4
import numpy as np
import scipy.linalg as la
import os
import sys
import time
from itertools import permutations

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from fock import *
from tools import *
from Hamiltonian import *

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

    # CAS assuming nbeta = nalpha

    def compute_CAS(self, active_space ='',nfrozen=0, nvirtual=0):

        # Read active spac. Format: sequence of letters ordered
        # according to orbital energies. Legend:
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
            raise NameError("Invalid active space. Please check the number of basis functions")
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

        self.ref = Det(a = ('1'*self.ndocc + '0'*self.nvir), \
                       b = ('1'*self.ndocc + '0'*self.nvir))
        print("Number of active orbitals: {}".format(n_ac_orb))
        print("Number of active electrons: {}".format(2*n_ac_elec_pair))

        # GENERATE DETERMINANTS

        print("Generating excitations")
        perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))

        self.determinants = []
        progress = 0
        file = sys.stdout
        for p1 in perms:
            for p2 in perms:
                self.determinants.append(Det(a=template_space.format(*p1), \
                                             b=template_space.format(*p2)))
            progress += 1
            showout(progress, len(perms), 50, "Generating Determinants: ", file)
        file.write('\n')
        file.flush()
        print("Number of determinants: {}".format(len(self.determinants)))

        # COMPUTE HAMILTONIAN MATRIX

        #print("Generating Hamiltonian Matrix")
        H = get_H(self.determinants, self.h, self.Vint, v = True, t = True)

        # DIAGONALIZE HAMILTONIAN MATRIX

        print("Diagonalizing Hamiltonian Matrix")
        t0 = time.time()
        E, Ccas = la.eigh(H)
        tf = time.time()
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Electronic Energy:")
        self.E = E[0]
        self.C0 = Ccas[:,0]
        self.Ecas = E[0]

        return self.Ecas

if __name__ == '__main__':
        
    # Input Geometry    
    
    mol = psi4.geometry("""
        0 1
        F 
        H 1 1.0
        symmetry c1
    """)

    #mol = psi4.geometry("""
    #    0 1
    #    
    #    Be 1 1.0
    #    symmetry c1
    #""")
    
    #mol = psi4.geometry("""
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
    
    basis = 'sto-3g'
    
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
    Enuc = mol.nuclear_repulsion_energy()
    
    Ecas = CAS.compute_CAS('aaaaaa')
    print("\nCAS Energy: {:<15.10f} ".format(Ecas+Enuc) + emoji('whale'))
    print("Time required: {} seconds.".format(time.time()-t))

