import numpy as np
import scipy.linalg as la
import os
import sys
import copy
import time
import psi4
import scipy.linalg as sp
from itertools import permutations

file_dir = os.path.dirname('../../../Aux/')
sys.path.append(file_dir)

from fock import Det
from Htools import *

def compute(OEI, ERI, nelec, ndocc, nvir, verb = False, ISE = False):

    def printif(x):
        if verb:
            print(x)

    ############### STEP 2 ###############
    ######  Generate Determinants  #######

    # Produce a reference determinant

    ref = Det(a = ('1'*ndocc + '0'*nvir), \
              b = ('1'*ndocc + '0'*nvir))

    perms = set(permutations('1'*ndocc + '0'*nvir))

    determinants = []
    progress = 0
    for p1 in perms:
        for p2 in perms:
            d = Det(a = ('{}'*(ndocc+nvir)).format(*p1), \
                    b = ('{}'*(ndocc+nvir)).format(*p2))
            if ref - d <= 4:
                determinants.append(d)
        progress += 1
    
    printif("Done.\n")
    print(len(determinants))

    # Construct the Hamiltonian Matrix
    # Note: Input for two electron integral must be using Chemists' notation

    H = get_H(determinants, OEI, ERI, v = verb, t = verb)

    # Diagonalize the Hamiltonian Matrix
    if not ISE: 
        printif("Diagonalizing Hamiltonian Matrix\n")

        t = time.time()

        # Matrices smaller than 100 x 100 will be fully diagonalize using eigh
        printif('Using Numpy Linear Algebra')
        E, Ccas = la.eigh(H)
        Ecas = E[0]
        Ccas = Ccas[:,0]

        printif('Diagonalization time: {}'.format(time.time()-t))

        return Ecas, Ccas, ref, determinants
    else: 
        printif("Diagonalizing Inverse Hamiltonian Matrix\n")

        t = time.time()

        printif('Inverting...\n')
        H = np.linalg.inv(H)
        E, Ccas = la.eigh(H)
        Ecas = max(E)
        Ccas = Ccas[:,0]

        printif('Diagonalization time: {}'.format(time.time()-t))

        return 1/Ecas, Ccas, ref, determinants
         
def CISD(ISE = False):

    print('\n --------- CASCI STARTED --------- \n')

    # Number of molecular orbitals is determined from the size of One-electron integral
    scf_e, wfn = psi4.energy('scf', return_wfn = True)
    nelec = wfn.nalpha() + wfn.nbeta()
    C = wfn.Ca()
    ndocc = wfn.doccpi()[0]
    nmo = wfn.nmo()
    nvir = nmo - ndocc
    eps = np.asarray(wfn.epsilon_a())
    nbf = C.shape[0]
    Vnuc = wfn.molecule().nuclear_repulsion_energy()

    print("Number of Electrons:            {}".format(nelec))
    print("Number of Basis Functions:      {}".format(nbf))
    print("Number of Molecular Orbitals:   {}".format(nmo))
    print("Number of Doubly ocuppied MOs:  {}\n".format(ndocc))

    # Build integrals from Psi4 MINTS
    
    print("Converting atomic integrals to MO integrals...")
    t = time.time()
    mints = psi4.core.MintsHelper(wfn.basisset())
    ERI = np.asarray(mints.mo_eri(C, C, C, C))
    OEI = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    OEI = np.einsum('up,vq,uv->pq', C, C, OEI)
    print("Completed in {} seconds!".format(time.time()-t))

    Ecas, Ccas, ref, determinants = compute(OEI, ERI, nelec, ndocc, nvir, verb=True, ISE = ISE)

    print('Final CISD energy: {:<15.10f}'.format(Ecas+Vnuc))

