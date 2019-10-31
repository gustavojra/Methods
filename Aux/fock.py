import numpy as np

# This module allows the constrction of objects that represent determinants for Configuration Interaction computations

class Det:

    # a and b inputs are strings for the occupancy of alpha and beta orbitals
    # For example a = '11100', b = '11100' is a determinant with all electrons in the lowest energy state
    # Now, the determinant a = '11100', b = '11010' represents a singly excited determinant.

    def __init__(self, a='', b='', ref = 'none', sq = False):

        # Strings are saved as the integer they represent. The order is inverted such that less memory is used for a given string

        self.alpha = int(a[::-1], 2)       
        self.beta  = int(b[::-1], 2)
        self.order = len(a)
    
        # If a reference was given attribute a sign to the determinants
        if sq:
            lcre = sorted(self.exclusive(ref))[::-1]
            lanh = sorted(ref.exclusive(self))
            molde = ref.copy()
            for lorb in lanh:
                molde.anh(lorb)
            for lorb in lcre:
                molde.cre(lorb)
            if molde == self:
                self.order = molde.order
            else:
                raise NameError('Error creating determinant:\n {s}\n Under reference:\n {s}'.format(str(self), str(ref)))

    def __str__(self):
        
        # Create a string representation of the determinant. Mostly used for debugging 

        out = '---------------\n'
        out += 'Alpha: ' + np.binary_repr(self.alpha, width=abs(self.order))[::-1]
        out += '\n'    
        out += 'Beta:  ' + np.binary_repr(self.beta, width=abs(self.order))[::-1]
        if self.order > 0:
            out += '\n' + '+'
        else:
            out += '\n' + '-'
        out += '\n---------------'
        return out

    def alpha_list(self):
        
        # Returns a list representing alpha electrons. For example: '11100' -> [1, 1, 1, 0, 0]
        # Note: output has left to right ordering

        return np.array([int(x) for x in list(np.binary_repr(self.alpha, width=abs(self.order)))])[::-1]

    def beta_list(self):

        # Returns a list representing beta electrons. For example: '11100' -> [1, 1, 1, 0, 0]
        # Note: output has left to right ordering

        return np.array([int(x) for x in list(np.binary_repr(self.beta, width=abs(self.order)))])[::-1]

    def alpha_string(self):

        # Returns a string representing alpha electrons
        # Note: output has right to left ordering (as it is stored)

        return np.binary_repr(self.alpha, width=abs(self.order))

    def beta_string(self):

        # Returns a string representing beta electrons
        # Note: output has right to left ordering (as it is stored)

        return np.binary_repr(self.beta, width=abs(self.order))

    def alpha_beta_string(self):

        # Returns a string representing both alpha and beta electrons
        # Note: output has right to left ordering (as it is stored)

        return self.beta_string() + self.alpha_string()

    def __eq__(self, other):

        # Compare if two Dets are the same.

        if self.alpha == other.alpha and self.beta == other.beta:
            return True
        else:
            return False

    def __sub__(self, other,v=False):

        # Subtracting two determinants yields the number of different orbitals between them.
        # The operations is commutative
        # Note that in this formulation 1 excitation => 2 different orbitals

        a = bin(self.alpha ^ other.alpha).count("1")
        b = bin(self.beta ^ other.beta).count("1")
        return a + b

    def exclusive(self, other):

        # Return orbitals occupied in the first det but not in the second
        # The index returned is positive for alpha electrons and negative for beta electrons

        out = []
        a = self.alpha_list() - other.alpha_list()
        for i in np.where(a == 1)[0]:
            out.append([i,0])
        b = self.beta_list() - other.beta_list()
        for i in np.where(b == 1)[0]:
            out.append([i,1])
        return out

    def copy(self):

        # Returns a copy of itself.

        return Det(a = self.alpha_string()[::-1], b = self.beta_string()[::-1])

    def rmv_alpha(self, orb):

        # Remove an alpha electron with orbital index 'orb'

        self.alpha ^= (1 << orb)

    def rmv_beta(self, orb):

        # Remove a beta electron with orbital index 'orb'

        self.beta ^= (1 << orb)

    def add_alpha(self, orb):

        # Create an alpha electron with orbital index 'orb'

        self.alpha |= (1 <<  orb)

    def add_beta(self, orb):

        # Create a beta electron with orbital index 'orb'

        self.beta |= (1 << orb)

    def sign_dif2(self, another):

        # Determines the phase create when two orbitals are put in maximum coincidence.
        # Should be used for Dets that differ by only two orbitals

        det1 = int(self.alpha_beta_string(),2)
        det2 = int(another.alpha_beta_string(),2)
        x1 = det1 & (det1 ^ det2)
        x2 = det2 & (det1 ^ det2)
        l = min(x1,x2)
        u = max(x1,x2)
        p = 0
        while l < u:
            u = u >> 1
            if u & (det1 & det2):
                p += 1
        return (-1)**p
        det1 = np.array(self.alpha_list() + self.beta_list())
        det2 = np.array(another.alpha_list() + another.beta_list())
        x = det1 - det2
            
    def sign_dif4(self, another):

        # Determines the phase create when two orbitals are put in maximum coincidence.
        # Should be used for Dets that differ by four orbitals

        det1 = int(self.alpha_beta_string(),2)
        det2 = int(another.alpha_beta_string(),2)
        x1 = det1 & (det1 ^ det2)
        x2 = det2 & (det1 ^ det2)
        p = 0
        i = 1
        px1 = []
        px2 = []
        while i < max(x1,x2):
            if i & (det1 & det2):
                    p += 1
            if i & x1:
                px1.append(p)
            if i & x2:
                px2.append(p)
            i = i << 1
        p = abs(px1[0]-px2[0]) + abs(px1[1]-px2[1])
        return (-1)**p
        
    def phase(self, another):

        # Determines the phase create when two orbitals are put in maximum coincidence.
        # Just a wrapper around the two functions above
    
        s1 = self.order/abs(self.order)
        s2 = another.order/abs(another.order)
        if self - another == 2:
            return self.sign_dif2(another)*s1*s2
        if self - another == 4:
            return self.sign_dif4(another)*s1*s2
        else:
            return 0

    def sign_del_alpha(self, orb):
    
        # Determine the sign needed to delete the orbital 'orb' i.e. the sign when you apply an annihilation operator on this Det

        i = 1
        count = 0

        while i < (1 << orb):
            if self.alpha & i:
                count += 1
            i = i << 1

        return (-1)**count
            
    def sign_del_beta(self, orb):
    
        # Determine the sign needed to delete the orbital 'orb' i.e. the sign when you apply an annihilation operator on this Det

        i = 1
        count = self.alpha_list().sum()

        while i < (1 << orb):
            if self.beta & i:
                count += 1
            i = i << 1

        return (-1)**count
            
    def anh(self, lorb):
        
        # Anihilate the desired orbital. Changes the phase according to second quantization rules
        # Input: List [orbital_index, spin (0 = alpha, 1 = beta)]

        [o, s] = lorb
        
        if s == 0:
            p = self.sign_del_alpha(o)
            self.rmv_alpha(o)
        else:
            p = self.sign_del_beta(o)
            self.rmv_beta(o)

        self.order = p*self.order

    def cre(self, lorb):
        
        # Create the desired orbital. Changes the phase according to second quantization rules
        # Input: List [orbital_index, spin (0 = alpha, 1 = beta)]

        [o, s] = lorb
        
        if s == 0:
            p = self.sign_del_alpha(o)
            self.add_alpha(o)
        else:
            p = self.sign_del_beta(o)
            self.add_beta(o)

        self.order = p*self.order

