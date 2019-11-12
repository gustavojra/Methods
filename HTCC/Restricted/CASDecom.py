import numpy as np
import time

##############################################################################
##############################################################################
##            ____    _    ____  ____                                       ##
##           / ___|  / \  / ___||  _ \  ___  ___ ___  _ __ ___              ##
##          | |     / _ \ \___ \| | | |/ _ \/ __/ _ \| '_ ` _ \             ##
##          | |___ / ___ \ ___) | |_| |  __/ (_| (_) | | | | | |            ##
##           \____/_/   \_\____/|____/ \___|\___\___/|_| |_| |_|            ##
##                                                                          ##
##     Inputs:  List with CI coefficients (Ccas);                           ##
##              Corresponding determinants (determinants) as Det            ##
##              objects from fock module;                                   ##
##              Reference determinant (ref) as a Det object;                ##                
##              Active Space (active_space) where the determinats           ##
##              were generated, following format used in the CASCI module.  ##
##                                                                          ##
##     Outputs: Translated amplitudes from CASCI coeffients:                ##
##          - T1;                                                           ## 
##          - T2(alpha, beta -> alpha, beta);                               ##
##          - T3(alpha, beta, alpha -> alpha, beta, alpha);                 ##
##          - T4(alpha, beta, alpha, alpha -> alpha, beta, alpha, alpha);   ##
##          - T4(alpha, beta, alpha, beta -> alpha, beta, alpha, beta);     ##
##                                                                          ##
##############################################################################
##############################################################################

def CASDecom(Ccas, determinants, ref, active_space):

    # Get number of doubly occupied and virtual orbitals from the reference determinant

    ndocc = int(sum(ref.alpha_list()))
    nvir = abs(ref.order) - ndocc

    # Get the C0 coefficient, corresponding to the reference determinant

    C0 = Ccas[determinants.index(ref)]
    
    # Since CC requires a non zero coefficient for the reference, it is required that C0 be above a threshold

    if abs(C0) < 0.01:
        raise NameError('Leading Coefficient too small C0 = {}\n Restricted orbitals not appropriate.'.format(C0))

    # Normalize the CI vector with respect to the reference

    Ccas = Ccas/C0
    
    # Create the arrays for amplitudes, inititally they will be used to store CI coefficients then they will
    # be translated into CC amplitudes

    CAS_T1 = np.zeros([ndocc, nvir])
    CAS_T2 = np.zeros([ndocc, ndocc, nvir, nvir])
    CAS_T3 = np.zeros([ndocc, ndocc, ndocc, nvir, nvir, nvir])
    CAS_T4abaa = np.zeros([ndocc, ndocc, ndocc, ndocc, nvir, nvir, nvir, nvir])
    CAS_T4abab = np.zeros([ndocc, ndocc, ndocc, ndocc, nvir, nvir, nvir, nvir])

    # Runs through the determinants to classify them by excitation rank. Collect CI coefficient and excitation indexes.
    t = time.time()

    for det,ci in zip(determinants, Ccas):
        if det - ref == 2:

            # If it is a singly excited determinant both spins (i,a) should be the same. This is taken care of in the CASCI module
            # We only collect the alpha excitation (i alpha -> a alpha) and save it on C1

            i = ref.exclusive(det)
            if i[0] != []:
                i = i[0][0]
                a = det.exclusive(ref)[0][0] - ndocc
                CAS_T1[i,a] = ci

        if det - ref == 4:

            # For doubly excited determinants we collect the spin case where ij -> ab: alpha, beta -> alpha, beta
            # Since the Det objects are constructed such that i < j and a < b and alpha < beta the sign should be correct already

            [i,j] = ref.exclusive(det)
            if i != [] and j != []:
                [a,b] = det.exclusive(ref)
                i, j = i[0], j[0]
                a, b = a[0] - ndocc, b[0] - ndocc
                CAS_T2[i,j,a,b] = ci

        if det - ref == 6:

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
                CAS_T3[i,j,k,a,b,c] = ci
                CAS_T3[k,j,i,a,b,c] = -ci
                CAS_T3[i,j,k,c,b,a] = -ci
                CAS_T3[k,j,i,c,b,a] = ci

        if det - ref == 8:

            # For quadruply excited determinants we want two spin cases:
            #    - a,b,a,a -> a,b,a,a
            #    - a,b,a,b -> a,b,a,b
            # where a means alpha, and b means beta (hopefully this is obvious, but lets be clear, right?)
            # The default in CASCI is: a,a,b,b -> a,a,b,b and a,a,a,b -> a,a,a,b
            # However, similarly to the T3 case we need to do a even number of permutations to get the desired determinants.
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
                CAS_T4abaa[i,j,k,l,a,b,c,d] = ci
                CAS_T4abaa[i,j,k,l,c,b,d,a] = ci
                CAS_T4abaa[i,j,k,l,a,b,d,c] = -ci
                CAS_T4abaa[i,j,k,l,c,b,a,d] = -ci
                CAS_T4abaa[i,j,k,l,d,b,a,c] = ci
                CAS_T4abaa[i,j,k,l,d,b,c,a] = -ci
                CAS_T4abaa[k,j,i,l,c,b,d,a] = -ci
                CAS_T4abaa[k,j,i,l,a,b,d,c] = ci
                CAS_T4abaa[k,j,i,l,c,b,a,d] = ci
                CAS_T4abaa[k,j,i,l,a,b,c,d] = -ci
                CAS_T4abaa[k,j,i,l,d,b,a,c] = -ci
                CAS_T4abaa[k,j,i,l,d,b,c,a] = ci
                CAS_T4abaa[k,j,l,i,c,b,d,a] = ci
                CAS_T4abaa[k,j,l,i,a,b,d,c] = -ci
                CAS_T4abaa[k,j,l,i,c,b,a,d] = -ci
                CAS_T4abaa[k,j,l,i,a,b,c,d] = ci
                CAS_T4abaa[k,j,l,i,d,b,a,c] = ci
                CAS_T4abaa[k,j,l,i,d,b,c,a] = -ci
                CAS_T4abaa[l,j,k,i,c,b,d,a] = -ci
                CAS_T4abaa[l,j,k,i,a,b,d,c] = ci
                CAS_T4abaa[l,j,k,i,c,b,a,d] = ci
                CAS_T4abaa[l,j,k,i,a,b,c,d] = -ci
                CAS_T4abaa[l,j,k,i,d,b,a,c] = -ci
                CAS_T4abaa[l,j,k,i,d,b,c,a] = ci
                CAS_T4abaa[l,j,i,k,c,b,d,a] = ci
                CAS_T4abaa[l,j,i,k,a,b,d,c] = -ci
                CAS_T4abaa[l,j,i,k,c,b,a,d] = -ci
                CAS_T4abaa[l,j,i,k,a,b,c,d] = ci
                CAS_T4abaa[l,j,i,k,d,b,a,c] = ci
                CAS_T4abaa[l,j,i,k,d,b,c,a] = -ci
                CAS_T4abaa[i,j,l,k,c,b,d,a] = -ci
                CAS_T4abaa[i,j,l,k,a,b,d,c] = ci
                CAS_T4abaa[i,j,l,k,c,b,a,d] = ci
                CAS_T4abaa[i,j,l,k,a,b,c,d] = -ci
                CAS_T4abaa[i,j,l,k,d,b,a,c] = -ci
                CAS_T4abaa[i,j,l,k,d,b,c,a] = ci

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
                CAS_T4abab[i,j,k,l,a,b,c,d] = ci
                CAS_T4abab[i,j,k,l,a,d,c,b] = -ci
                CAS_T4abab[i,j,k,l,c,d,a,b] = ci
                CAS_T4abab[i,j,k,l,c,b,a,d] = -ci
                CAS_T4abab[k,j,i,l,a,d,c,b] = ci
                CAS_T4abab[k,j,i,l,a,b,c,d] = -ci
                CAS_T4abab[k,j,i,l,c,d,a,b] = -ci
                CAS_T4abab[k,j,i,l,c,b,a,d] = ci
                CAS_T4abab[k,l,i,j,a,d,c,b] = -ci
                CAS_T4abab[k,l,i,j,a,b,c,d] = ci
                CAS_T4abab[k,l,i,j,c,d,a,b] = ci
                CAS_T4abab[k,l,i,j,c,b,a,d] = -ci
                CAS_T4abab[i,l,k,j,a,d,c,b] = ci
                CAS_T4abab[i,l,k,j,a,b,c,d] = -ci
                CAS_T4abab[i,l,k,j,c,d,a,b] = -ci
                CAS_T4abab[i,l,k,j,c,b,a,d] = ci

    print('Time for collection: {}'.format(time.time() - t))
    # Translate CI coefficients into CC amplitudes
    
    print('Translating CI coefficients into CC amplitudes...\n')
    
    # Singles: already done
    
    # Doubles
    CAS_T2 = CAS_T2 - np.einsum('ia,jb-> ijab', CAS_T1, CAS_T1)
    T2aa = CAS_T2 - np.einsum('ijab -> jiab', CAS_T2)
    
    # Triples
    
    CAS_T3 += - np.einsum('ia,jkbc -> ijkabc', CAS_T1, CAS_T2) \
                      + np.einsum('ic,kjab -> ijkabc', CAS_T1, CAS_T2) \
                      + np.einsum('ka,ijcb -> ijkabc', CAS_T1, CAS_T2) \
                      - np.einsum('kc,ijab -> ijkabc', CAS_T1, CAS_T2) \
                      - np.einsum('jb,ikac -> ijkabc', CAS_T1, T2aa)
    
    CAS_T3 += - np.einsum('ia,jb,kc -> ijkabc', CAS_T1, CAS_T1, CAS_T1) \
                      + np.einsum('ic,jb,ka -> ijkabc', CAS_T1, CAS_T1, CAS_T1) 
    
    CAS_T3aaa = CAS_T3 - np.einsum('ijkabc -> ikjabc', CAS_T3) - np.einsum('ijkabc -> jikabc', CAS_T3)
    print('Translating quadruples...')
    # Quadruples
    
    print('Spin case (ABAA -> ABAA)')
    ## First case abaa -> abaa
    
    CAS_T4abaa = CAS_T4abaa
    
    print('... T1 * T3...')
    ### T1 * T3 terms
    
    CAS_T4abaa += - np.einsum('ia, kjlcbd -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('ic, kjlabd -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('id, kjlabc -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('jb, iklacd -> ijklabcd', CAS_T1, CAS_T3aaa) \
                       + np.einsum('ka, ijlcbd -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('kc, ijlabd -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('kd, ijlabc -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('la, ijkcbd -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('lc, ijkabd -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('ld, ijkabc -> ijklabcd', CAS_T1, CAS_T3)
    
    print('... T2 * T2...')
    ### T2 * T2 terms
    
    CAS_T4abaa += - np.einsum('ijab, klcd -> ijklabcd', CAS_T2, T2aa) \
                       + np.einsum('ijcb, klad -> ijklabcd', CAS_T2, T2aa) \
                       - np.einsum('ijdb, klac -> ijklabcd', CAS_T2, T2aa) \
                       - np.einsum('ljdb, ikac -> ijklabcd', CAS_T2, T2aa) \
                       + np.einsum('ljcb, ikad -> ijklabcd', CAS_T2, T2aa) \
                       - np.einsum('ljab, ikcd -> ijklabcd', CAS_T2, T2aa) \
                       + np.einsum('kjdb, ilac -> ijklabcd', CAS_T2, T2aa) \
                       - np.einsum('kjcb, ilad -> ijklabcd', CAS_T2, T2aa) \
                       + np.einsum('kjab, ilcd -> ijklabcd', CAS_T2, T2aa) 
    
    print('... T1 * T1 * T2...')
    ### T1 * T1 * T2 terms
    
    CAS_T4abaa += - np.einsum('jb,ia,klcd -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       + np.einsum('jb,ic,klad -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       - np.einsum('jb,id,klac -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       + np.einsum('jb,ka,ilcd -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       - np.einsum('jb,kc,ilad -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       + np.einsum('jb,kd,ilac -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       - np.einsum('jb,la,ikcd -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       + np.einsum('jb,lc,ikad -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       - np.einsum('jb,ld,ikac -> ijklabcd', CAS_T1, CAS_T1, T2aa) \
                       - np.einsum('ia,kc,ljdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ia,kd,ljcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ia,lc,kjdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ia,ld,kjcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ic,ka,ljdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ic,kd,ljab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ic,la,kjdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ic,ld,kjab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('id,ka,ljcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('id,kc,ljab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('id,la,kjcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('id,lc,kjab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ka,lc,ijdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ka,ld,ijcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('kc,la,ijdb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('kc,ld,ijab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('kd,la,ijcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('kd,lc,ijab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2)
    
    print('... T1 * T1 * T1 * T1...')
    ### T1 * T1 * T1 * T1 terms
    
    CAS_T4abaa += - np.einsum('jb, ia, kc, ld -> ijkabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1) \
                       + np.einsum('jb, ia, kd, lc -> ijkabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1) \
                       + np.einsum('jb, ic, ka, ld -> ijkabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1) \
                       - np.einsum('jb, ic, kd, la -> ijkabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1) \
                       - np.einsum('jb, id, ka, lc -> ijkabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1) \
                       + np.einsum('jb, id, kc, la -> ijkabcd', CAS_T1,CAS_T1,CAS_T1,CAS_T1) 
    
    print('Spin case (ABAB -> ABAB)')
    ## Second case: abab -> abab
    
    CAS_T4_abab = CAS_T4abab
    
    ### T1 * T3 terms
    print('... T1 * T3...')
    
    CAS_T4abab += - np.einsum('ia, jklbcd -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('ic, jklbad -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('jb, ilkadc -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('jd, ilkabc -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('ka, jilbcd -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('kc, jilbad -> ijklabcd', CAS_T1, CAS_T3) \
                       + np.einsum('lb, ijkadc -> ijklabcd', CAS_T1, CAS_T3) \
                       - np.einsum('ld, ijkabc -> ijklabcd', CAS_T1, CAS_T3)
    
    print('... T2 * T2...')
    ### T2 * T2 terms
    
    CAS_T4abab += - np.einsum('ijab, klcd -> ijklabcd', CAS_T2, CAS_T2) \
                       + np.einsum('ijad, klcb -> ijklabcd', CAS_T2, CAS_T2) \
                       + np.einsum('ijcb, klad -> ijklabcd', CAS_T2, CAS_T2) \
                       - np.einsum('ijcd, klab -> ijklabcd', CAS_T2, CAS_T2) \
                       - np.einsum('ikac, jlbd -> ijklabcd', T2aa, T2aa)               \
                       + np.einsum('ilab, kjcd -> ijklabcd', CAS_T2, CAS_T2) \
                       - np.einsum('ilad, jkbc -> ijklabcd', CAS_T2, CAS_T2) \
                       - np.einsum('ilcb, kjad -> ijklabcd', CAS_T2, CAS_T2) \
                       + np.einsum('ilcd, kjab -> ijklabcd', CAS_T2, CAS_T2) 
    
    print('... T1 * T1 * T3...')
    ### T1 * T1 * T2 terms
    
    CAS_T4abab += - np.einsum('ia,jb,klcd -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ia,jd,klcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ia,kc,jlbd -> ijklabcd', CAS_T1, CAS_T1, T2aa)        \
                       + np.einsum('ia,lb,kjcd -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ia,ld,jkbc -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ic,jb,klad -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('ic,jd,klab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ic,ka,jlbd -> ijklabcd', CAS_T1, CAS_T1, T2aa)        \
                       - np.einsum('ic,lb,kjad -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ic,ld,kjab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('jb,ka,ilcd -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('jb,kc,ilad -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('jb,ld,ikac -> ijklabcd', CAS_T1, CAS_T1, T2aa)        \
                       - np.einsum('jd,ka,ilcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('jd,kc,ilab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('jd,lb,ikac -> ijklabcd', CAS_T1, CAS_T1, T2aa)        \
                       - np.einsum('ka,lb,ijcd -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('ka,ld,ijcb -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       + np.einsum('kc,lb,ijad -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) \
                       - np.einsum('kc,ld,ijab -> ijklabcd', CAS_T1, CAS_T1, CAS_T2) 
    
    print('... T1 * T1 * T1 * T1...')
    ### T1 * T1 * T1 * T1 terms
    
    CAS_T4abab += - np.einsum('ia,jb,kc,ld -> ijklabcd', CAS_T1, CAS_T1, CAS_T1, CAS_T1) \
                       + np.einsum('ia,jd,kc,lb -> ijklabcd', CAS_T1, CAS_T1, CAS_T1, CAS_T1) \
                       + np.einsum('ic,jb,ka,ld -> ijklabcd', CAS_T1, CAS_T1, CAS_T1, CAS_T1) \
                       - np.einsum('ic,jd,ka,lb -> ijklabcd', CAS_T1, CAS_T1, CAS_T1, CAS_T1)

    return CAS_T1, CAS_T2, CAS_T3, CAS_T4abab, CAS_T4abaa
