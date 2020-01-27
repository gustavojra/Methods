import psi4
import os
import sys
import numpy as np
import time
import copy

sys.path.append('../../Aux')

from tools import *


class CCSD:

    def __init__(self, wfn, CC_CONV=6, CC_MAXITER=50, E_CONV=8):
        self.Ehf = wfn.energy() 
        self.nalpha = wfn.nalpha() 
        self.nbeta = wfn.nbeta()
        self.Ca = wfn.Ca()
        self.Cb = wfn.Cb()
        self.fda = np.asarray(wfn.epsilon_a())
        self.fdb = np.asarray(wfn.epsilon_b())
        self.nmo = wfn.nmo()
        self.Vnuc = wfn.molecule().nuclear_repulsion_energy()

        print("Number of alpha electrons:   {}".format(self.nalpha))
        print("Number of beta electrons:    {}".format(self.nbeta))
        print("Number of MOs:               {}".format(self.nmo))

        print("\n Transforming integrals...")

        mints = psi4.core.MintsHelper(wfn.basisset())
        # One electron integral
        h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        ## Alpha
        ah = np.einsum('up,vq,uv->pq', self.Ca, self.Ca, h)
        ## Beta
        bh = np.einsum('up,vq,uv->pq', self.Cb, self.Cb, h)
        Vaaaa = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))
        Vbbbb = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))
        Vaabb = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))
        Vbbaa = np.asarray(mints.mo_eri(Cb, Cb, Ca, Ca))

        fa = ah + np.einsum('pqkk->pq', Vaaaa - Vaaaa.swapaxes(


        
        

        self.compute()


        
        
