import numpy as np
import scipy.linalg as la
import os
import sys
import copy
import time
from itertools import permutations

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

import tools as tool
from fock import Det
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
##              ○ Number of spatial orbitals (nmo);                         ##
##              ○ Number of electrons (nelec);                              ##
##              ○ One-electron Integral (OEI);                              ##
##              ○ Two-electron Integral, using physicists notation (TEI).   ##
##                                                                          ##
##     Outputs: ○ CAS energy (Ecas), that is, lowest eigenvalue             ##
##              of the Hamiltonian Matrix                                   ##
##              Warning: This do not include Nuclei repulsion;              ##
##              ○ Ground state CI coefficients (Ccas);                      ##
##              ○ Reference determinant (ref) as a Det object               ##
##              from the fock module;                                       ##
##              ○ List of determinants (determinants) as Det objects        ##
##              generated in the given active space;                        ##
##              ○ Keyword (return_as) for returning Active Space string     ##
##              after processed (active_space), False by default.           ##
##                                                                          ##
##############################################################################
##############################################################################
         
def CASCI(active_space, nmo, nelec, OEI, TEI, show_prog = False, davidson = False, return_as = False):

    # Read active space and determine number of active electrons. 
    # Standard Format: sequence of letters ordered
    # according to orbital energies. 
    # Legend:
    # o = frozen doubly occupied orbital
    # a = active orbital
    # u = frozen unnocupied orbital 
    # Alternative formats:
    # - List with active orbitals indexes
    # - 'full' for a FCI computation
    # - 'none' for a a empty active space, that is, no CAS is performed

    print('\n --------- CASCI STARTED --------- \n')

    ndocc = int(nelec/2)
    nvir = nmo - ndocc

    # For a list input "active list". If the orbital is in the active list it is considered active 'a'
    # if not, it will be considered doubly occupied or unnocupied depending on the reference correspondence.


    if type(active_space) == list:
        indexes = copy.deepcopy(active_space)
        active_space = ''
        hf_string = 'o'*ndocc + 'u'*nvir
        for i in range(nmo):
            if i in indexes:
                active_space += 'a'
            else:
                active_space += hf_string[i]

    # 'none' produces an empty space, thus no CAS will be performed.
    # Well, in reality it will be performed, but the Hamiltonian matrix
    # will be 1x1.

    if active_space == 'none':
        active_space = 'o'*ndocc + 'u'*nvir

    # Setting active_space = 'full' calls Full CI

    if active_space == 'full':
        active_space = 'a'*nmo

    # Creating a template string. Active orbitals are represented by {}
    # Occupied orbitals are 1, and unnocupied 0. 
    # For example: 11{}{}{}000 represents a system with 8 orbitals where the two lowest ones are frozen (doubly occupied)
    # and the three highest one are frozen (unnocupied). There are 3 active orbitals.
    # The number of active electron pairs is obtained as the number of electron pairs minus the number of doubly occupied orbitals ('o')
    
    template_space = ''
    n_ac_orb = 0
    n_ac_elec_pair = int(nelec/2)
    print("Reading Active space")
    if active_space == 'full':
        active_space = 'a'*nmo
    if len(active_space) != nmo:
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

    ref = Det(a = ('1'*ndocc + '0'*nvir), \
              b = ('1'*ndocc + '0'*nvir))

    print("Number of active orbitals: {}".format(n_ac_orb))
    print("Number of active electrons: {}\n".format(2*n_ac_elec_pair))

    # Produces a list of active electrons in active orbitals and get all permutations of it.
    # For example. Say we have a template 11{}{}{}000 as in the example above. If the system contains 6 electrons
    # we have one pair of active electrons. The list of active electrons will look like '100' and the permutations
    # will be generated (as lists): ['1', '0', '0'], ['0', '1', '0'], and ['0', '0', '1'].

    print("Generating excitations...")
    perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))
    print("Done.\n")

    # Each permutation is then merged with the template above to generate a string used ot create a Determinant object
    # For example. 11{}{}{}000 is combine with ['0','1', '0'] to produce an alpha/beta string 11010000.
    # These strings are then combined to form various determinants

    # The option 'sq' set as true will make the Det object to compare the newly create determinant with the reference.
    # This is done to provide a sign/phase to the determinant consistent with second quantization operators

    determinants = []
    progress = 0
    file = sys.stdout
    for p1 in perms:
        for p2 in perms:
            determinants.append(Det(a = template_space.format(*p1), \
                                    b = template_space.format(*p2), \
                                               ref = ref,sq=True))
        progress += 1
        tool.showout(progress, len(perms), 50, "Generating Determinants: ", file)
    file.write('\n')
    file.flush()
    print("Number of determinants:                   {}".format(len(determinants)))
    nelements = int((len(determinants)*(len(determinants)-1))/2) + len(determinants)
    print("Number of matrix elements to be computed: {}".format(nelements))

    # Construct the Hamiltonian Matrix
    # Note: Input for two electron integral must be using Chemists' notation

    H = get_H(determinants, OEI, TEI.swapaxes(1,2), v = True, t = True)

    # Diagonalize the Hamiltonian Matrix
    print("Diagonalizing Hamiltonian Matrix")

    t = time.time()

    # Matrices smaller than 100 x 100 will be fully diagonalize using eigh
    if davidson and len(H) > 100:
        print('Using Davidson solver')
        Ecas, Ccas = Davidson(H, kmax=100)
    else:
        print('Using Numpy Linear Algebra')
        E, Ccas = la.eigh(H)
        Ecas = E[0]
        Ccas = Ccas[:,0]

    print('Diagonalization time: {}'.format(time.time()-t))
    if return_as:
        return Ecas, Ccas, ref, determinants, active_space

    return Ecas, Ccas, ref, determinants
