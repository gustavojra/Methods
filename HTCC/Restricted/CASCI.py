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
         
def CASCI(active_space, nelec, OEI, TEI, davidson = False, return_as = False):

    print('\n --------- CASCI STARTED --------- \n')

    # Number of molecular orbitals is determined from the size of One-electron integral
    nmo = len(OEI)
    ndocc = int(nelec/2)
    nvir = nmo - ndocc

    if nmo != len(TEI):
        raise NameError('One and Two electron integrals have different dimentions')

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

    print("Number of active spatial orbitals:  {}".format(n_ac_orb))
    print("Number of active electrons:         {}\n".format(2*n_ac_elec_pair))

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

    print("Generating excitations...\n")
    perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))
    ndets = len(perms)**2
    nelements = int((ndets*(ndets+1))/2)
    print("Number of determinants:                   {}".format(ndets))
    print("Number of unique matrix elements:         {}\n".format(nelements))

    determinants = []
    progress = 0
    for p1 in perms:
        for p2 in perms:
            determinants.append(Det(a = template_space.format(*p1), \
                                    b = template_space.format(*p2), \
                                               ref = ref, sq=True))
        progress += 1

    print("Done.\n")

    # Construct the Hamiltonian Matrix
    # Note: Input for two electron integral must be using Chemists' notation

    H = get_H(determinants, OEI, TEI.swapaxes(1,2), v = True, t = True)

    # Diagonalize the Hamiltonian Matrix
    print("Diagonalizing Hamiltonian Matrix\n")

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
