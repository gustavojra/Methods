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

import tools as tool
from fock import Det
from CCSD import CCSD
from CCD import CCD
from davidson import Davidson
from Htools import *

##############################################################################
##############################################################################
##       ____    _    ____   ____ ___   __  __           _       _          ##
##      / ___|  / \  / ___| / ___|_ _| |  \/  | ___   __| |_   _| | ___     ##
##     | |     / _ \ \___ \| |    | |  | |\/| |/ _ \ / _` | | | | |/ _ \    ##
##     | |___ / ___ \ ___) | |___ | |  | |  | | (_) | (_| | |_| | |  __/    ##
##      \____/_/   \_\____/ \____|___| |_|  |_|\___/ \__,_|\__,_|_|\___|    ##
##                                                                          ##
##     • Performs a Complete Active Space Configuration Interaction         ##
##       i.e. a Full CI computation within a spacified active space         ##
##                                                                          ##
##     Inputs:  ○ Active Space (active_space);                              ##
##              ○ Number of electrons (nelec);                              ##
##              ○ One-electron Integral (OEI);                              ##
##              ○ Two-electron Integral, using physicists notation (TEI).   ##
##              ○ Keyword (return_as) for returning Active Space string     ##
##              after processed (active_space), False by default.           ##
##                                                                          ##
##     Outputs: ○ CAS energy (Ecas), that is, lowest eigenvalue             ##
##              of the Hamiltonian Matrix                                   ##
##              Warning: This do not include Nuclei repulsion;              ##
##              ○ Ground state CI coefficients (Ccas);                      ##
##              ○ Reference determinant (ref) as a Det object               ##
##              from the fock module;                                       ##
##              ○ List of determinants (determinants) as Det objects        ##
##              generated in the given active space;                        ##
##                                                                          ##
##############################################################################
##############################################################################

def inner(u,v, S_ao):
    return np.einsum('u,v,uv->', u, v, S_ao)

def proj(u, v, S_ao):
    return (inner(v,u, S_ao)/inner(u,u,S_ao))*u

def normalize(u, S_ao):
    return u/np.sqrt(inner(u,u, S_ao))

def gram_schmidt(C, S_ao):
    Cout = []
    for V in C.T:
        newV = V
        for Vn in Cout:
            newV -= proj(Vn,V, S_ao)
        newV = normalize(newV, S_ao)
        Cout.append(newV)  
    return np.array(Cout).T

def unitary(C, T1, ndocc, nvir, mints):
    X = np.block([
                 [ np.zeros([ndocc, ndocc]), np.zeros_like(T1)],
                 [ T1.T, np.zeros([nvir, nvir])]])
    
    U = sp.expm(X - X.T)
    C = C.np
    C = C.dot(U)
    C = psi4.core.Matrix.from_array(C)

    return C

def thouless(C, T1, ndocc, nvir, mints):
    o = slice(0,ndocc)
    v = slice(ndocc, ndocc+nvir)
    C = C.np
    Cocc = copy.deepcopy(C[:,o])
    Cvir = copy.deepcopy(C[:,v])

    C[:,o] = Cocc + np.einsum('ua,ia->ui', Cvir, T1)
    C[:,v] = Cvir - np.einsum('ui,ia->ua', Cocc, T1)

    S_ao = np.array(mints.ao_overlap())
    C = gram_schmidt(C, S_ao)
    C = psi4.core.Matrix.from_array(C)
    return C

def get_fock(h, V, ndocc):
    o = slice(0,ndocc)
    fo = h + 2*np.einsum('pqii->pq', V[:,:,o,o]) - np.einsum('piqi->pq', V[:,o,:,o])
    fd = np.array(fo.diagonal())
    np.fill_diagonal(fo, 0)
    return fd, fo

def overlap(C, mints):
    S = np.array(mints.ao_overlap())
    Sorb = np.einsum('up,vq,uv->pq', C, C, S)
    return Sorb

def compute(active_space, OEI, ERI, nelec, ndocc, nvir, verb = False):

    def printif(x):
        if verb:
            print(x)

    # Creating a template string. Active orbitals are represented by {}
    # Occupied orbitals are 1, and unnocupied 0. 
    # For example: 11{}{}{}000 represents a system with 8 orbitals where the two lowest ones are frozen (doubly occupied)
    # and the three highest ones are frozen (unnocupied). Thus, there are 3 active orbitals.
    
    template_space = active_space.replace('o', '1')
    template_space = template_space.replace('u', '0')
    template_space = template_space.replace('a', '{}')
    n_ac_orb = active_space.count('a')
    n_ac_elec_pair = int(nelec/2) - active_space.count('o')

    if n_ac_elec_pair < 0:
            raise NameError("{} frozen electrons for {} total electrons".format(2*active_space.count('o'), nelec))

    printif("Number of active spatial orbitals:  {}".format(n_ac_orb))
    printif("Number of active electrons:         {}\n".format(2*n_ac_elec_pair))

    ############### STEP 2 ###############
    ######  Generate Determinants  #######

    # Produce a reference determinant

    ref = Det(a = ('1'*ndocc + '0'*nvir), \
              b = ('1'*ndocc + '0'*nvir))

    # Produces a list of active electrons in active orbitals and get all permutations of it.
    # For example. Say we have a template 11{}{}{}000 as in the example above. If the system contains 6 electrons
    # we have one pair of active electrons. The list of active electrons will look like '100' and the permutations
    # will be generated (as lists): ['1', '0', '0'], ['0', '1', '0'], and ['0', '0', '1'].

    # Each permutation is then merged with the template above to generate a string used ot create a Determinant object
    # For example. 11{}{}{}000 is combine with ['0','1', '0'] to produce an alpha/beta string 11010000.
    # These strings are then combined to form various determinants

    # The option 'sq' set as true will make the Det object to compare the newly create determinant with the reference.
    # This is done to provide a sign/phase to the determinant consistent with second quantization operators

    printif("Generating excitations...\n")
    perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))
    ndets = len(perms)**2
    nelements = int((ndets*(ndets+1))/2)
    printif("Number of determinants:                   {}".format(ndets))
    printif("Number of unique matrix elements:         {}\n".format(nelements))

    determinants = []
    progress = 0
    for p1 in perms:
        for p2 in perms:
            determinants.append(Det(a = template_space.format(*p1), \
                                    b = template_space.format(*p2), \
                                               ref = ref, sq=True))
        progress += 1

    printif("Done.\n")

    # Construct the Hamiltonian Matrix
    # Note: Input for two electron integral must be using Chemists' notation

    H = get_H(determinants, OEI, ERI, v = verb, t = verb)

    # Diagonalize the Hamiltonian Matrix
    printif("Diagonalizing Hamiltonian Matrix\n")

    t = time.time()

    # Matrices smaller than 100 x 100 will be fully diagonalize using eigh
    printif('Using Numpy Linear Algebra')
    E, Ccas = la.eigh(H)
    Ecas = E[0]
    Ccas = Ccas[:,0]

    printif('Diagonalization time: {}'.format(time.time()-t))

    return Ecas, Ccas, ref, determinants
         
def CASCI(active_space, davidson = False, rot = False, rot_method = 1, ccsd = False, ccd = False):

    print('\n --------- CASCI STARTED --------- \n')

    # Number of molecular orbitals is determined from the size of One-electron integral
    inp_acs = copy.deepcopy(active_space)
    scf_e, wfn = psi4.energy('scf', return_wfn = True)
    print(scf_e)
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

    ############### STEP 1 ###############
    ########  Read Active Space  #########

    ## The active_space must be a string containing the letters 'o', 'a' and 'u':
    ##  'o' represents frozen doubly occupied orbitals;
    ##  'a' represents active orbitals;
    ##  'u' represents frozen unoccupied orbitals.
    ## However, the active_space can also be given in three alternative ways:
    ## active_space = 'full': for a full CI computation;
    ## active_space = 'none': for a emputy active space (output energy is SCF energy);
    ## active_space = list
    ## In the latter case, the list must contain indexes of the active orbitals. For example:
    ## active_space = 'oooaaauuaaauuu' can be more conveniently written as active_space = [3,4,5,8,9,10]
    ## Note that, the first orbital has index 0

    print('Processing active space')

    if active_space == 'full':
        active_space = 'a'*nmo

    if active_space == 'none':
        active_space = 'o'*ndocc + 'u'*nvir

    if type(active_space) == list:
        indexes = copy.deepcopy(active_space)
        active_space = ''
        hf_string = 'o'*ndocc + 'u'*nvir
        for i in range(nmo):
            if i in indexes:
                active_space += 'a'
            else:
                active_space += hf_string[i]

    # Check if active space size is consistem with given integrals
    if len(active_space) != nmo:
        raise NameError('Active Space size is {} than the number of molecular orbitals'.format('greater' if len(active_space) > nmo else 'smaller'))

    Ecas, Ccas, ref, determinants = compute(active_space, OEI, ERI, nelec, ndocc, nvir, verb=True)

    print('First CAS Energy', Ecas+Vnuc)

    T1 = np.zeros([ndocc, nvir])
    C0 = Ccas[list(determinants).index(ref)]
    Ccas = Ccas/C0

    for det,ci in zip(determinants, Ccas):
        if det - ref == 2:

            # If it is a singly excited determinant both spins (i,a) should be the same. This is taken care of in the CASCI module
            # We only collect the alpha excitation (i alpha -> a alpha) and save it on C1

            i = ref.exclusive(det)
            if i[0] != []:
                i = i[0][0]
                a = det.exclusive(ref)[0][0] - ndocc
                T1[i,a] = ci

    if rot:

        print('Starting Orbital rotations\n')
        print('='*30)
        ite = 1

    while rot and np.max(abs(T1)) > 1.e-12:

        if rot_method == 1:
            print('Unitary rotation')
            C = unitary(C, T1, ndocc, nvir, mints)
        elif rot_method == 2:
            print('Thouless theorem')
            C = thouless(C, T1, ndocc, nvir, mints)
        else:
            raise NameError('Invalid rot type')

        S = overlap(C.np, mints)
        Str = np.trace(S)
        Soffd = np.sum(S) - Str

        print('Iteration {}'.format(ite))

        ERI = np.asarray(mints.mo_eri(C, C, C, C))
        OEI = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        OEI = np.einsum('up,vq,uv->pq', C, C, OEI)
        
        Ecas, Ccas, ref, determinants = compute(active_space, OEI, ERI, nelec, ndocc, nvir)
        occs = slice(0,ndocc)
        scf_e = 2*np.einsum('ii->', OEI[occs,occs]) + 2*np.einsum('iijj->', ERI[occs,occs,occs,occs]) - np.einsum('ijij->', ERI[occs,occs,occs,occs]) + Vnuc

        print('CAS energy: {:<5.10f}'.format(Ecas + Vnuc))
        print('Ref energy: {:<5.10f}'.format(scf_e))

        T1 = np.zeros([ndocc, nvir])
        C0 = Ccas[list(determinants).index(ref)]
        print('C0:         {:<5.10f}'.format(C0))
        Ccas = Ccas/C0
        print('S trace:    {:<4.2f}'.format(Str))
        print('S off diag: {:<4.2f}'.format(Soffd))

        for det,ci in zip(determinants, Ccas):
            if det - ref == 2:

                # If it is a singly excited determinant both spins (i,a) should be the same. This is taken care of in the CASCI module
                # We only collect the alpha excitation (i alpha -> a alpha) and save it on C1

                i = ref.exclusive(det)
                if i[0] != []:
                    i = i[0][0]
                    a = det.exclusive(ref)[0][0] - ndocc
                    T1[i,a] = ci

        tool.printmatrix(T1)
        ite += 1
        print('T1 amplitudes')
        printmatrix(T1)
        print('='*30)

    printmatrix(C.np)
    fd, fo =get_fock(OEI, ERI, ndocc)
    
    if ccsd:
        CCSD(ERI.swapaxes(1,2), fd, fo, ndocc, nvir, scf_e)

    if ccd:
        CCD(ERI.swapaxes(1,2), fd, fo, ndocc, nvir, scf_e)
