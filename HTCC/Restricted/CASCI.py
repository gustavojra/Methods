import numpy as np
import scipy.linalg as la
import os
import sys
import copy
from itertools import permutations

file_dir = os.path.dirname('../../Aux/')
sys.path.append(file_dir)

import tools as tool
from fock import Det
from Hamiltonian import *

##############################################################################
##############################################################################
##       ____    _    ____   ____ ___   __  __           _       _          ##
##      / ___|  / \  / ___| / ___|_ _| |  \/  | ___   __| |_   _| | ___     ##
##     | |     / _ \ \___ \| |    | |  | |\/| |/ _ \ / _` | | | | |/ _ \    ##
##     | |___ / ___ \ ___) | |___ | |  | |  | | (_) | (_| | |_| | |  __/    ##
##      \____/_/   \_\____/ \____|___| |_|  |_|\___/ \__,_|\__,_|_|\___|    ##
##                                                                          ##
##     Inputs:  Active Space (active_space);                                ##
##              Number of spatial orbitals (nmo);                           ##
##              Number of electrons (nelec);                                ##
##              One-electron Integral (OEI);                                ##
##              Two-electron Integral, using physicists notation (TEI).     ##
##                                                                          ##
##     Outputs: CAS energy (Ecas), that is, lowest eigenvalue               ##
##              of the Hamiltonian Matrix                                   ##
##              Warning: This do not include Nuclei repulsion;              ##
##              Ground state CI coefficients (Ccas);                        ##
##              Reference determinant (ref) as a Det object                 ##
##              from the fock module;                                       ##
##              List of determinants (determinants) as Det objects          ##
##              generated in the given active space;                        ##
##              Active Space after processed (active_space).                ##
##                                                                          ##
##############################################################################
##############################################################################
         
def CASCI(active_space, nmo, nelec, OEI, TEI, show_prog = False):

    # Read active space and determine number of active electrons. 
    # Format: sequence of letters ordered
    # according to orbital energies. 
    # Legend:
    # o = frozen doubly occupied orbital
    # a = active orbital
    # u = frozen unnocupied orbital 

    ndocc = int(nelec/2)
    nvir = nmo - ndocc

    if type(active_space) == list:
        indexes = copy.deepcopy(active_space)
        active_space = ''
        hf_string = 'o'*ndocc + 'u'*nvir
        for i in range(nmo):
            if i in indexes:
                active_space += 'a'
            else:
                active_space += hf_string[i]

    # none is none

    if active_space == 'none':
        active_space = 'o'*ndocc + 'u'*nvir

    # Setting active_space = 'full' calls Full CI

    if active_space == 'full':
        active_space = 'a'*nmo

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

    # Use permutations to generate strings that will represent excited determinants

    print("Generating excitations...")
    perms = set(permutations('1'*n_ac_elec_pair + '0'*(n_ac_orb - n_ac_elec_pair)))
    print("Done.\n")

    # Use the strings to generate Det objects. Use second quantization to attribute signs 

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
    print("Number of determinants: {}".format(len(determinants)))

    # Construct the Hamiltonian Matrix
    # Note: Input for two electron integral must be using Chemists' notation

    H = get_H(determinants, OEI, TEI.swapaxes(1,2), v = True, t = True)

    # Diagonalize the Hamiltonian Matrix

    print("Diagonalizing Hamiltonian Matrix")
    E, Ccas = la.eigh(H)
    Ecas = E[0]
    Ccas = Ccas[:,0]
    return Ecas, Ccas, ref, determinants, active_space
