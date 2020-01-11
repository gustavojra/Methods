import psi4
import numpy as np

##############################################################################
##############################################################################
##            ____    _    ____  ____                                       ##
##           / ___|  / \  / ___||  _ \  ___  ___ ___  _ __ ___              ##
##          | |     / _ \ \___ \| | | |/ _ \/ __/ _ \| '_ ` _ \             ##
##          | |___ / ___ \ ___) | |_| |  __/ (_| (_) | | | | | |            ##
##           \____/_/   \_\____/|____/ \___|\___\___/|_| |_| |_|            ##
##                                                                          ##
##     • Performs Cluster Decomposition of CI coefficients                  ##
##                                                                          ##
##     Inputs: ○ List with CI coefficients (Ccas);                          ##
##             ○ Corresponding determinants (determinants) as Det           ##
##               objects from fock module;                                  ##
##             ○ Reference determinant (ref) as a Det object;               ##                
##             ○ Active Space following CASCI format;                       ##
##             ○ Keyword for returning T3 (return_t3), true by default.     ## 
##             ○ Keyword for returning T4 (return_t4), true by default.     ## 
##                                                                          ##
##     Outputs: Translated amplitudes from CASCI coeffients:                ##
##          ○ T1;                                                           ## 
##          ○ T2(alpha, beta -> alpha, beta);                               ##
##          ○ T3(alpha, beta, alpha -> alpha, beta, alpha);                 ##
##          ○ T4(alpha, beta, alpha, alpha -> alpha, beta, alpha, alpha);   ##
##          ○ T4(alpha, beta, alpha, beta -> alpha, beta, alpha, beta).     ##
##                                                                          ##
##############################################################################
##############################################################################

print_to_output = True

def printout(x):
    if print_to_output:
        psi4.core.print_out(x)
        psi4.core.print_out('\n')
    else:
        print(x)

def CASDecom(Ccas, determinants, ref, fdocc = 0, fvir = 0, return_t3 = True, return_t4 = True):

    ############### STEP 1 ###############
    ##########  Initial Workup  ##########

    # If T4 is requested T3 must also be.  
    if return_t4 and not return_t3:
        raise NameError('Cannot return T4 without T3, please set return_t3 = True. Could I do that automatically? Yes, but this is pedagogical.')

    printout('\n --------- CASDecom STARTED --------- \n')

    # Get number of doubly occupied and virtual orbitals from the reference determinant

    ndocc = int(sum(ref.alpha_list()))
    nvir = abs(ref.order) - ndocc

    # Get the C0 coefficient, corresponding to the reference determinant

    C0 = Ccas[np.where(np.array(determinants) == ref)[0][0]]
    maxC = np.max(abs(np.array(Ccas)))
    
    # Since CC requires a non zero coefficient for the reference, it is required that C0 be above a threshold

    if abs(C0) < 0.01:
        raise NameError('Leading Coefficient too small C0 = {}\n Restricted orbitals not appropriate.'.format(C0))

    printout('Abs HF C0 value:    {:<5.5f}'.format(abs(C0)))
    printout('Max CI coef found:  {:<5.5f}'.format(abs(maxC)))

    # Normalize the CI vector with respect to the reference

    Ccas = np.array(Ccas)/C0
    
    # Create the arrays for amplitudes, inititally they will be used to store CI coefficients then they will
    # be translated into CC amplitudes

    T1 = np.zeros([ndocc, nvir])
    T2 = np.zeros([ndocc, ndocc, nvir, nvir])
    if return_t3:
        T3 = np.zeros([ndocc, ndocc, ndocc, nvir, nvir, nvir])
    if return_t4:
        T4abaa = np.zeros([ndocc, ndocc, ndocc, ndocc, nvir, nvir, nvir, nvir])
        T4abab = np.zeros([ndocc, ndocc, ndocc, ndocc, nvir, nvir, nvir, nvir])

    # Runs through the determinants to classify them by excitation rank. Collect CI coefficient and excitation indexes.

    ############### STEP 2 ###############
    ####  CI coefficients collections ####

    printout('Collecting CI coefficients from CASCI eigenvector')
    for det,ci in zip(determinants, Ccas):
        if det - ref == 2:

            # If it is a singly excited determinant both spins (i,a) should be the same. This is taken care of in the CASCI module
            # We only collect the alpha excitation (i alpha -> a alpha) and save it on C1

            i = ref.exclusive(det)
            if i[0] != []:
                i = i[0][0]
                a = det.exclusive(ref)[0][0] - ndocc
                T1[i,a] = ci

        if det - ref == 4:

            # For doubly excited determinants we collect the spin case where ij -> ab: alpha, beta -> alpha, beta
            # Since the Det objects are constructed such that i < j and a < b and alpha < beta the sign should be correct already

            [i,j] = ref.exclusive(det)
            if i != [] and j != []:
                [a,b] = det.exclusive(ref)
                i, j = i[0], j[0]
                a, b = a[0] - ndocc, b[0] - ndocc
                T2[i,j,a,b] = ci

        if det - ref == 6 and return_t3:

            # For triply excited determinants we want to collect the case ijk -> abc: alpha, beta, alpha -> alpha, beta, alpha
            # By default, CASCI used alpha, alpha, beta -> alpha, alpha, beta where i < j < k and a < b < c. We need to swap two
            # indexes to get the desired determinat, that means that there is no sign change.
            # We also need to consider permutations of i,k and a,c. Although they map to the same determinant in the CASCI, we need 
            # to have these numbers in the T3 array for the CC framework since we are working with unrestricted arrays on einsum.

            [ik, j] = ref.exclusive(det)
            if len(ik) == 2:
                [ac, b] = det.exclusive(ref)
                ik = sorted(ik)
                i, j, k = ik[0], j[0], ik[1]
                ac = sorted(ac)
                a, b, c = ac[0] - ndocc, b[0] - ndocc, ac[1] - ndocc
                # Include all permutations P(ik)P(ac)
                T3[i,j,k,a,b,c] =  ci
                T3[k,j,i,a,b,c] = -ci
                T3[i,j,k,c,b,a] = -ci
                T3[k,j,i,c,b,a] =  ci

        if det - ref == 8 and return_t4:

            # For quadruply excited determinants we want two spin cases:
            #    - a,b,a,a -> a,b,a,a
            #    - a,b,a,b -> a,b,a,b
            # where 'a' means alpha, and 'b' means beta.
            # The default in CASCI is: a,a,b,b -> a,a,b,b and a,a,a,b -> a,a,a,b
            # However, similarly to the T3 case, we need to do a even number of permutations to get the desired determinants.
            # Thus, no sign adjustment is needed, except for the permutations. For each case we have:
            #    - a,b,a,a case: P(ik)P(il)P(kl)P(ac)P(ad)P(cd)
            #    - a,b,a,b case: P(ik)P(jl)P(ac)P(bd)

            [alphas, betas] = ref.exclusive(det)
            if len(alphas) == 3:
                # This is spin case: a,b,a,a
                alphas = sorted(alphas)
                i,j,k,l = alphas[0], betas[0], alphas[1], alphas[2]
                [acd, b] = det.exclusive(ref)
                acd = sorted(acd)
                a, b, c, d = acd[0] - ndocc, b[0] - ndocc, acd[1] - ndocc, acd[2] - ndocc
                # Include all permutations P(ik)P(il)P(kl)P(ac)P(ad)P(cd)
                T4abaa[i,j,k,l,a,b,c,d] =  ci
                T4abaa[i,j,k,l,c,b,d,a] =  ci
                T4abaa[i,j,k,l,a,b,d,c] = -ci
                T4abaa[i,j,k,l,c,b,a,d] = -ci
                T4abaa[i,j,k,l,d,b,a,c] =  ci
                T4abaa[i,j,k,l,d,b,c,a] = -ci
                T4abaa[k,j,i,l,c,b,d,a] = -ci
                T4abaa[k,j,i,l,a,b,d,c] =  ci
                T4abaa[k,j,i,l,c,b,a,d] =  ci
                T4abaa[k,j,i,l,a,b,c,d] = -ci
                T4abaa[k,j,i,l,d,b,a,c] = -ci
                T4abaa[k,j,i,l,d,b,c,a] =  ci
                T4abaa[k,j,l,i,c,b,d,a] =  ci
                T4abaa[k,j,l,i,a,b,d,c] = -ci
                T4abaa[k,j,l,i,c,b,a,d] = -ci
                T4abaa[k,j,l,i,a,b,c,d] =  ci
                T4abaa[k,j,l,i,d,b,a,c] =  ci
                T4abaa[k,j,l,i,d,b,c,a] = -ci
                T4abaa[l,j,k,i,c,b,d,a] = -ci
                T4abaa[l,j,k,i,a,b,d,c] =  ci
                T4abaa[l,j,k,i,c,b,a,d] =  ci
                T4abaa[l,j,k,i,a,b,c,d] = -ci
                T4abaa[l,j,k,i,d,b,a,c] = -ci
                T4abaa[l,j,k,i,d,b,c,a] =  ci
                T4abaa[l,j,i,k,c,b,d,a] =  ci
                T4abaa[l,j,i,k,a,b,d,c] = -ci
                T4abaa[l,j,i,k,c,b,a,d] = -ci
                T4abaa[l,j,i,k,a,b,c,d] =  ci
                T4abaa[l,j,i,k,d,b,a,c] =  ci
                T4abaa[l,j,i,k,d,b,c,a] = -ci
                T4abaa[i,j,l,k,c,b,d,a] = -ci
                T4abaa[i,j,l,k,a,b,d,c] =  ci
                T4abaa[i,j,l,k,c,b,a,d] =  ci
                T4abaa[i,j,l,k,a,b,c,d] = -ci
                T4abaa[i,j,l,k,d,b,a,c] = -ci
                T4abaa[i,j,l,k,d,b,c,a] =  ci

            if len(alphas) == 2:
                # This is spin case: a,b,a,b
                alphas = sorted(alphas)
                betas = sorted(betas)
                i,j,k,l = alphas[0], betas[0], alphas[1], betas[1]
                [ac, bd] = det.exclusive(ref)
                ac = sorted(ac)
                bd = sorted(bd)
                a, b, c, d = ac[0] - ndocc, bd[0] - ndocc, ac[1] - ndocc, bd[1] - ndocc
                # Include all permutations P(ik)P(jl)P(ac)P(bd)
                T4abab[i,j,k,l,a,b,c,d] =  ci
                T4abab[i,j,k,l,a,d,c,b] = -ci
                T4abab[i,j,k,l,c,d,a,b] =  ci
                T4abab[i,j,k,l,c,b,a,d] = -ci
                T4abab[k,j,i,l,a,d,c,b] =  ci
                T4abab[k,j,i,l,a,b,c,d] = -ci
                T4abab[k,j,i,l,c,d,a,b] = -ci
                T4abab[k,j,i,l,c,b,a,d] =  ci
                T4abab[k,l,i,j,a,d,c,b] = -ci
                T4abab[k,l,i,j,a,b,c,d] =  ci
                T4abab[k,l,i,j,c,d,a,b] =  ci
                T4abab[k,l,i,j,c,b,a,d] = -ci
                T4abab[i,l,k,j,a,d,c,b] =  ci
                T4abab[i,l,k,j,a,b,c,d] = -ci
                T4abab[i,l,k,j,c,d,a,b] = -ci
                T4abab[i,l,k,j,c,b,a,d] =  ci

    printout('Collection completed.\n')

    # Create slices of the big T arrays. For small active spaces, most of these arrays are going to be zeros. Therefore, we only need 
    # the active part of it for the next step. This reduces the cost of the following tensor contractions.

    h = slice(fdocc, ndocc)
    p = slice(0, nvir-fvir)

    CAS_T1     = T1[h,p]
    CAS_T2     = T2[h,h,p,p]

    if return_t3:
        CAS_T3     = T3[h,h,h,p,p,p]
    if return_t4:
        CAS_T4abaa = T4abaa[h,h,h,h,p,p,p,p]
        CAS_T4abab = T4abab[h,h,h,h,p,p,p,p]

    
    ############### STEP 3 ###############
    ######  Cluster Decomposition  #######

    printout('Cluster Decomposition Started')
    
    # Singles: equivalent to CI
    printout('   -> T1        done.')
    
    # Doubles
    CAS_T2 += - np.einsum('ia,jb-> ijab', CAS_T1, CAS_T1, optimize='optimal')

    printout('   -> T2        done.')

    if not return_t3:
        T1[h,p]                 =   CAS_T1     
        T2[h,h,p,p]             =   CAS_T2     
        printout('Decomposition completed.')
        printout('\n --------- CASDecom FINISHED ---------\n')
        return T1, T2

    ## Compute the spin case a,a -> a,a from the mixed spin case: Used for T3 and T4.
    T2aa = CAS_T2 - CAS_T2.transpose(1,0,2,3)

    # Triples
    ## Taking advantage of permutation symmetry when possible

    t1t2 = np.einsum('ia, jkbc -> ijkabc', CAS_T1, CAS_T2, optimize='optimal')
    t1t1 = np.einsum('ia,jb,kc -> ijkabc', CAS_T1, CAS_T1, CAS_T1, optimize='optimal')

    CAS_T3 += - t1t2                                                             \
              + t1t2.transpose(0,1,2,5,4,3)                                      \
              + t1t2.transpose(2,1,0,3,4,5)                                      \
              - t1t2.transpose(2,1,0,5,4,3)                                      \
              - np.einsum('jb,ikac -> ijkabc', CAS_T1, T2aa, optimize='optimal') \
              - t1t1                                                             \
              + t1t1.transpose(0,1,2,5,4,3)

    printout('   -> T3        done.')

    if not return_t4:
        T1[h,p]                 =   CAS_T1     
        T2[h,h,p,p]             =   CAS_T2     
        T3[h,h,h,p,p,p]         =   CAS_T3     
        printout('Decomposition completed.')
        printout('\n --------- CASDecom FINISHED ---------\n')
        return T1, T2, T3

    ### Compute the spin case a,a,a -> a,a,a from the mixed spin case: Used for T4
    CAS_T3aaa = CAS_T3 - CAS_T3.transpose(0,2,1,3,4,5) - CAS_T3.transpose(1,0,2,3,4,5)

    # Quadruples
    
    ## First case abaa -> abaa

    ### T1 * T3 terms
    
    t1t3a = np.einsum('ia,kjlcbd -> ijklabcd', CAS_T1, CAS_T3, optimize='optimal') 
    t1t3b = np.einsum('kd,ijlabc -> ijklabcd', CAS_T1, CAS_T3, optimize='optimal') 

    CAS_T4abaa += - t1t3a                                                                      \
                  + t1t3a.transpose(0,1,2,3,6,5,4,7)                                           \
                  - t1t3a.transpose(0,1,3,2,7,5,6,4)                                           \
                  - np.einsum('jb, iklacd -> ijklabcd', CAS_T1, CAS_T3aaa, optimize='optimal') \
                  + t1t3a.transpose(2,1,0,3,4,5,6,7)                                           \
                  - t1t3a.transpose(2,1,0,3,6,5,4,7)                                           \
                  + t1t3b                                                                      \
                  - t1t3a.transpose(3,1,2,0,4,5,7,6)                                           \
                  + t1t3b.transpose(0,1,3,2,4,5,7,6)                                           \
                  - t1t3b.transpose(0,1,3,2,4,5,6,7)
    
    ### T2 * T2 terms
    
    t2t2a = np.einsum('ijab, klcd -> ijklabcd', CAS_T2, T2aa, optimize='optimal')
    t2t2b = np.einsum('ljcb, kida -> ijklabcd', CAS_T2, T2aa, optimize='optimal')

    CAS_T4abaa += - t2t2a                            \
                  + t2t2a.transpose(0,1,2,3,6,5,4,7) \
                  - t2t2a.transpose(0,1,3,2,7,5,6,4) \
                  - t2t2a.transpose(3,1,2,0,7,5,6,4) \
                  + t2t2b                            \
                  - t2t2a.transpose(3,1,2,0,4,5,7,6) \
                  + t2t2b.transpose(0,1,3,2,4,5,7,6) \
                  - t2t2a.transpose(2,1,0,3,6,5,4,7) \
                  + t2t2a.transpose(2,1,0,3,4,5,6,7) 

    ### T1 * T1 * T2 terms
    
    t1t1t2a = np.einsum('ia, jb, klcd -> ijklabcd', CAS_T1, CAS_T1, T2aa, optimize='optimal')
    t1t1t2b = np.einsum('kd, jb, ilac -> ijklabcd', CAS_T1, CAS_T1, T2aa, optimize='optimal')
    t1t1t2c = np.einsum('ia, kc, ljdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2, optimize='optimal')
    t1t1t2d = np.einsum('ic, ld, kjab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2, optimize='optimal')

    CAS_T4abaa += - t1t1t2a                            \
                  + t1t1t2a.transpose(0,1,2,3,6,5,4,7) \
                  - t1t1t2a.transpose(0,1,3,2,7,5,6,4) \
                  + t1t1t2a.transpose(2,1,0,3,4,5,6,7) \
                  - t1t1t2a.transpose(2,1,0,3,6,5,4,7) \
                  + t1t1t2b                            \
                  - t1t1t2a.transpose(3,1,2,0,4,5,7,6) \
                  + t1t1t2b.transpose(0,1,3,2,4,5,7,6) \
                  - t1t1t2b.transpose(0,1,3,2,4,5,6,7) \
                  - t1t1t2c                            \
                  + t1t1t2c.transpose(0,1,2,3,4,5,7,6) \
                  + t1t1t2c.transpose(0,1,3,2,4,5,6,7) \
                  - t1t1t2c.transpose(0,1,3,2,4,5,7,6) \
                  + t1t1t2c.transpose(0,1,2,3,6,5,4,7) \
                  - t1t1t2c.transpose(2,1,0,3,7,5,6,4) \
                  - t1t1t2c.transpose(0,1,3,2,6,5,4,7) \
                  + t1t1t2d                            \
                  - t1t1t2c.transpose(2,1,0,3,4,5,7,6) \
                  + t1t1t2c.transpose(0,1,2,3,7,5,6,4) \
                  + t1t1t2d.transpose(3,1,2,0,6,5,4,7) \
                  - t1t1t2d.transpose(0,1,2,3,4,5,7,6) \
                  - t1t1t2c.transpose(3,1,2,0,6,5,4,7) \
                  + t1t1t2d.transpose(2,1,0,3,6,5,4,7) \
                  + t1t1t2c.transpose(3,1,2,0,4,5,6,7) \
                  - t1t1t2d.transpose(2,1,0,3,4,5,6,7) \
                  - t1t1t2c.transpose(3,1,2,0,4,5,7,6) \
                  + t1t1t2d.transpose(2,1,0,3,4,5,7,6) 

    ### T1 * T1 * T1 * T1 terms

    t1t1t1t1 = np.einsum('ia, jb, kc, ld -> ijklabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1, optimize='optimal') 

    CAS_T4abaa += - t1t1t1t1                            \
                  + t1t1t1t1.transpose(0,1,2,3,4,5,7,6) \
                  + t1t1t1t1.transpose(0,1,2,3,6,5,4,7) \
                  - t1t1t1t1.transpose(0,1,3,2,6,5,4,7) \
                  - t1t1t1t1.transpose(0,1,3,2,7,5,6,4) \
                  + t1t1t1t1.transpose(0,1,2,3,7,5,6,4) 

    printout('   -> T4 (ABAA) done.')

    ## Second case: abab -> abab
    
    ### T1 * T3 terms
    
    t1t3a = np.einsum('ia, jklbcd -> ijklabcd', CAS_T1, CAS_T3, optimize='optimal')
    t1t3b = np.einsum('jd, ilkabc -> ijklabcd', CAS_T1, CAS_T3, optimize='optimal')
    CAS_T4abab += - t1t3a                               \
                  + t1t3a.transpose(0,1,2,3,6,5,4,7)    \
                  - t1t3a.transpose(1,0,3,2,5,4,7,6)    \
                  + t1t3b                               \
                  + t1t3a.transpose(2,1,0,3,4,5,6,7)    \
                  - t1t3a.transpose(2,1,0,3,6,5,4,7)    \
                  + t1t3b.transpose(0,3,2,1,4,7,6,5)    \
                  - t1t3b.transpose(0,3,2,1,4,5,6,7)
    
    ### T2 * T2 terms

    t2t2 = np.einsum('ijab, klcd -> ijklabcd', CAS_T2, CAS_T2, optimize='optimal')

    CAS_T4abab += - t2t2                                                                    \
                  + t2t2.transpose(0,1,2,3,4,7,6,5)                                         \
                  + t2t2.transpose(0,1,2,3,6,5,4,7)                                         \
                  - t2t2.transpose(0,1,2,3,6,7,4,5)                                         \
                  + t2t2.transpose(0,3,2,1,4,5,6,7)                                         \
                  - t2t2.transpose(0,3,2,1,6,5,4,7)                                         \
                  + t2t2.transpose(0,3,2,1,6,7,4,5)                                         \
                  - np.einsum('ilad, jkbc -> ijklabcd', CAS_T2, CAS_T2, optimize='optimal') \
                  - np.einsum('ikac, jlbd -> ijklabcd', T2aa, T2aa, optimize='optimal') 
    
    ### T1 * T1 * T2 terms

    t1t1t2a = np.einsum('ia,jb,klcd -> ijklabcd', CAS_T1, CAS_T1, CAS_T2, optimize='optimal') 
    t1t1t2b = np.einsum('jb,ka,ilcd -> ijklabcd', CAS_T1, CAS_T1, CAS_T2, optimize='optimal')

    CAS_T4abab += - t1t1t2a                                                                         \
                  + t1t1t2a.transpose(0,1,2,3,4,7,6,5)                                              \
                  + t1t1t2a.transpose(0,3,2,1,4,5,6,7)                                              \
                  + t1t1t2a.transpose(0,1,2,3,6,5,4,7)                                              \
                  - t1t1t2a.transpose(0,1,2,3,6,7,4,5)                                              \
                  - t1t1t2a.transpose(0,3,2,1,6,5,4,7)                                              \
                  + t1t1t2a.transpose(0,3,2,1,6,7,4,5)                                              \
                  - t1t1t2a.transpose(2,3,0,1,4,5,6,7)                                              \
                  + t1t1t2a.transpose(2,3,0,1,4,7,6,5)                                              \
                  + t1t1t2a.transpose(2,3,0,1,6,5,4,7)                                              \
                  - t1t1t2a.transpose(2,3,0,1,6,7,4,5)                                              \
                  + t1t1t2b                                                                         \
                  - t1t1t2b.transpose(0,1,2,3,6,5,4,7)                                              \
                  - t1t1t2b.transpose(0,1,2,3,4,7,6,5)                                              \
                  + t1t1t2b.transpose(0,1,2,3,6,7,4,5)                                              \
                  - t1t1t2b.transpose(2,3,0,1,4,7,6,5)                                              \
                  - np.einsum('ia,kc,jlbd -> ijklabcd', CAS_T1, CAS_T1, T2aa, optimize='optimal')   \
                  + np.einsum('ic,ka,jlbd -> ijklabcd', CAS_T1, CAS_T1, T2aa, optimize='optimal')   \
                  - np.einsum('jb,ld,ikac -> ijklabcd', CAS_T1, CAS_T1, T2aa, optimize='optimal')   \
                  + np.einsum('jd,lb,ikac -> ijklabcd', CAS_T1, CAS_T1, T2aa, optimize='optimal')   

    ### T1 * T1 * T1 * T1 terms
    
    t1t1t1t1 = np.einsum('ia,jb,kc,ld -> ijklabcd', CAS_T1, CAS_T1, CAS_T1, CAS_T1, optimize='optimal')

    CAS_T4abab += - t1t1t1t1                            \
                  + t1t1t1t1.transpose(0,1,2,3,4,7,6,5) \
                  + t1t1t1t1.transpose(0,1,2,3,6,5,4,7) \
                  - t1t1t1t1.transpose(0,1,2,3,6,7,4,5) 

    printout('   -> T4 (ABAB) done.')
    printout('Decomposition completed.')

    ### These next few lines are just for safety:
    ### Making sure that the modifications in the slices were traferred properly to the original arrays. Note
    ### that this will not be the case if 'Sliced_Array += X' is substituted by 'Sliced_Array = Sliced_Array + X'.

    T1[h,p]                 =   CAS_T1     
    T2[h,h,p,p]             =   CAS_T2     
    T3[h,h,h,p,p,p]         =   CAS_T3     
    T4abaa[h,h,h,h,p,p,p,p] =   CAS_T4abaa 
    T4abab[h,h,h,h,p,p,p,p] =   CAS_T4abab 

    printout('\n --------- CASDecom FINISHED ---------\n')

    return T1, T2, T3, T4abab, T4abaa
