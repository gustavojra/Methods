from tools import *
from fock import *
import numpy as np
import time
import sys

def mask4(M, m1, m2, m3, m4):
    return M[m1,:,:,:][:,m2,:,:][:,:,m3,:][:,:,:,m4]

def H_dif0(det1, molint1, molint2):
    # mask
    alphas = det1.alpha_list()
    betas = det1.beta_list()

    hcore = np.einsum('mm,m->', molint1, alphas) + np.einsum('mm,m->', molint1, betas)

    t0 = time.time()
    # Compute J for all combinations of m n being alpha or beta
    JK  = np.einsum('mmnn, m, n', molint2, alphas, alphas, optimize = 'optimal')\
        + np.einsum('mmnn, m, n', molint2, betas, betas, optimize = 'optimal')  \
        + np.einsum('mmnn, m, n', molint2, alphas, betas, optimize = 'optimal') \
        + np.einsum('mmnn, m, n', molint2, betas, alphas, optimize = 'optimal')
    # For K m and n have to have the same spin, thus only two cases are considered
    JK -= np.einsum('mnnm, m, n', molint2, alphas, alphas, optimize = 'optimal')
    JK -= np.einsum('mnnm, m, n', molint2, betas, betas, optimize = 'optimal')
    reft = time.time()-t0
    print('Conventional: {}  Time: {}  Relative Time: {}'.format(JK, reft, 1.0))

    t0 = time.time()
    a = np.array(alphas) == 1
    b = np.array(betas) == 1
    # Compute J for all combinations of m n being alpha or beta
    JK2  = np.einsum('mmnn ->', mask4(molint2,a,a,a,a), optimize = 'optimal')\
         + np.einsum('mmnn ->', mask4(molint2,b,b,b,b), optimize = 'optimal')  \
         + np.einsum('mmnn ->', mask4(molint2,a,a,b,b), optimize = 'optimal') \
         + np.einsum('mmnn ->', mask4(molint2,b,b,a,a), optimize = 'optimal')
    # For K m and n have to have the same spin, thus only two cases are considered
    JK2 -= np.einsum('mnnm ->', mask4(molint2,a,a,a,a), optimize = 'optimal')
    JK2 -= np.einsum('mnnm ->', mask4(molint2,b,b,b,b), optimize = 'optimal')
    t2 = time.time() - t0
    print('Mask:         {}  Time: {}  Relative Time: {}'.format(JK2, t2,(t2)/reft))
    return 0.5 * JK + hcore
    
    
def H_dif4(det1, det2, molint1, molint2):
    phase = det1.phase(det2)
    [alp1, bet1] = det1.exclusive(det2)
    [alp2, bet2] = det2.exclusive(det1)
    if len(alp1) != len(alp2):
        return 0
    [o1,o2,o3,o4] = alp1 + bet1 + alp2 + bet2
    if len(alp1) == 1:
        return phase * (molint2[o1,o3,o2,o4])
    else:
        return phase * (molint2[o1,o3,o2,o4] - molint2[o1,o4,o2,o3])

    #[[o1, s1], [o2, s2]] = det1.exclusive(det2)
    #[[o3, s3], [o4, s4]] = det2.exclusive(det1)
    #if s1 == s3 and s2 == s4:
    #    J = molint2[o1, o3, o2, o4] 
    #else:
    #    J = 0
    #if s1 == s4 and s2 == s3:
    #    K = molint2[o1, o4, o2, o3]
    #else:
    #    K = 0
    #return phase * (J - K)

def H_dif2(det1, det2, molint1, molint2):
    # Use exclusive to return a list of alpha and beta orbitals present in the first det, but no in the second 
    [alp1, bet1] = det1.exclusive(det2)
    [alp2, bet2] = det2.exclusive(det1)
 #   if len(alp1) != len(alp2):  # Check if the different orbitals have same spin  # DONT THINK THIS IS NECESSARY
 #       return 0
    phase = det1.phase(det2)
    [o1, o2] = alp1 + bet1 + alp2 + bet2
    # For J, (mp|nn), n can have any spin. Two cases are considered then. Obs: det1.occ or det2.occ would yield the same result. When n = m or p J - K = 0
    J = np.einsum('nn, n->', molint2[o1,o2], det1.alpha_list()) + np.einsum('nn, n->', molint2[o1,o2], det1.beta_list()) 
    if len(alp1) > 0:
        K = np.einsum('nn, n->', molint2.swapaxes(1,3)[o1,o2], det1.alpha_list())
    else:
        K = np.einsum('nn, n->', molint2.swapaxes(1,3)[o1,o2], det1.beta_list())
    return phase * (molint1[o1,o2] + J - K)


# FUNCTION: Given a list of determinants, compute the Hamiltonian matrix

def get_H(dets, molint1, molint2, v = False, t = False):
        l = len(dets)
        H = np.zeros((l,l))
        t0 = time.time()
        file = sys.stdout
        td = time.time()
        for i,d1 in enumerate(dets):
            H[i,i] = H_dif0(d1, molint1, molint2)
        for i,d1 in enumerate(dets):
            for j,d2 in enumerate(dets):
                if j >= i:
                    break
                dif = d1 - d2
                if dif == 4:
                    H[i,j] = H_dif4(d1, d2, molint1, molint2)
                    H[j,i] = H[i,j]
                elif dif == 2:
                    H[i,j] = H_dif2(d1, d2, molint1, molint2)
                    H[j,i] = H[i,j]
            if v: showout(i+1, l, 50, "Generating Hamiltonian Matrix: ", file)
        file.write('\n')
        file.flush()
        if t:
            print("Completed. Time needed: {}".format(time.time() - t0))
        return H

def H_diag(a_mask, b_mask, OEI, TEI):
    Anti = TEI - TEI.swapaxes(1,2)
    m = a_mask == b_mask
    m2 = a_mask != b_mask
    out = 2*np.einsum('mm ->', OEI[m,:][:,m]) + np.einsum('mm ->', OEI[m2,:][:,m2])
    out += 0.5*np.einsum('mmnn ->', Anti[m,:,:,:][:,m,:,:][:,:,m,:][:,:,:,m])
    out += np.einsum('mmnn ->', Anti[m2,:,:,:][:,m2,:,:][:,:,m,:][:,:,:,m])
    out += np.einsum('mmnn ->', Anti[m2,:,:,:][:,m2,:,:][:,:,m2,:][:,:,:,m2])
    return out
    
def Hgen(inp):
    det1 = inp[0]
    det2 = inp[1]
    one  = inp[2]
    two  = inp[3] 
    dif = det1 - det2
    if dif > 4:
        return 0
    elif dif == 4:
       return H_dif4(det1, det2, one, two) 
    elif dif == 2:
       return H_dif2(det1, det2, one, two) 
    elif dif == 0:
       return H_dif0(det1, one, two) 

