import numpy as np
from fock import *

class cc_wfn:

    def __init__(self, ref, ci_coef, ci_dets):

        # Read in inputs
        self.ref = ref

        # Get C0 based on the given reference
        C0 = ci_coef[dets.index(self.ref)]

        # Normalize CI coefficients 
        self.ci_coef = ci_coef/C0

        # Save single, double, triple and quadruple excited dets
        # Info save as [Coeficient, Determinant]
        self.singles = []
        self.doubles = []
        self.triples = []
        self.quadrup = []

        # Subtraction of det objects gives the number of different orbitals
        for i,x in enumerate(dets):
            dif = self - x
            if dif == 2:
                self.singles.append([ci_coef[i], x])

            if dif == 4:
                self.doubles.append([ci_coef[i], x])

            if dif == 6:
                self.triples.append([ci_coef[i], x])

            if dif == 8:
                self.quadrup.append([ci_coef[i], x])

        # Correct CI coefficients to a second quatization approach

        for i,x in enumerate(self.singles):
            [[Corb,Cspin]] = x[1].exclusive(self.ref)
            [[Aorb,Aspin]] = self.ref.exclusive(x[0])
