import numpy as np
from fock import *

class cc_wfn:

    def __init__(self, ci_coef, ci_dets):
        self.ci_coef = ci_coef
        self.dets = ci_dets
        self.cc_elements = []
        self.C0 = ci_coef.max()

