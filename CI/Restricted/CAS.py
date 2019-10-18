import psi4
import os
import sys
import numpy as np
import scipy.linalg as la
import time
import copy
from multiprocessing import Pool
from itertools import permutations

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

from tools import *
from fock import *
from Hamiltonian import *

np.set_printoptions(suppress=True)

class CAS:

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

        # Use the strings to generate Det objects 

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

        # Construct the Hamiltonian Matrix
        # Note: Input for two electron integral must be using Chemists' notation

        # Serial
        #H = get_H(self.determinants, self.h, self.Vint.swapaxes(1,2), v = True, t = True)

        # Parallel

        t = time.time()

        l = len(self.determinants)
        H = np.zeros((l,l))
        X = H_generator(self.h, self.Vint.swapaxes(1,2), self.determinants)
        p = Pool(6)
        a,b = np.triu_indices(l)
        indices = list(zip(a,b))
        out = p.map(X.gen, indices)
        H[a,b] = out
        print('time parallel: {}'.format(time.time()-t))

        #Diagonalize the Hamiltonian Matrix

        print("Diagonalizing Hamiltonian Matrix")
        t0 = time.time()
        E, Ccas = la.eigh(H)
        tf = time.time()
        self.Ecas = E[0] + self.Vnuc
        self.Ccas = Ccas[:,0]
        print("Completed. Time needed: {}".format(tf - t0))
        print("CAS Energy: {:<5.10f}".format(self.Ecas))

