import numpy as np

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

    ndocc = sum(ref.alpha_list())
    nvir = abs(ref.order) - ndocc

    # Get the C0 coefficient, corresponding to the reference determinant

    C0 = Ccas[determinants.index(ref)]
    
    # Since CC requires a non zero coefficient for the reference, it is required that C0 be above a threshold

    if abs(C0) < 0.01:
        raise NameError('Leading Coefficient too small C0 = {}\n Restricted orbitals not appropriate.'.format(C0))

    # Normalize the CI vector with respect to the reference

    Ccas = Ccas/C0
    
    # Slice the CI list and determinants in excitation cases

    C1dets = []
    C2dets = []
    C3dets = []
    C4dets = []
    C1 = []
    C2 = []
    C3 = []
    C4 = []

    for d,c in zip(determinants, Ccas):
        if d - ref == 2:
            C1.append(c)
            C1dets.append(d)
        if d - ref == 4:
            C2.append(c)
            C2dets.append(d)
        if d - ref == 6:
            C3.append(c)
            C3dets.append(d)
        if d - ref == 8:
            C4.append(c)
            C4dets.append(d)
            
    CAS_T1 = np.zeros([ndocc, nvir])
    
    CAS_holes = []
    for i,x in enumerate(active_space[0:ndocc]):
        if x == 'a':
            CAS_holes.append(i)
    
    CAS_particles = []
    for i,x in enumerate(active_space[ndocc:]):
        if x == 'a':
            CAS_particles.append(i)
    
    CAS_T2 = np.zeros([ndocc, ndocc, nvir, nvir])
    
    CAS_T3aba = np.zeros([ndocc, ndocc, ndocc, nvir, nvir, nvir])
    
    CAS_T4abaa = np.zeros([ndocc, ndocc, ndocc, ndocc, nvir, nvir, nvir, nvir])
    CAS_T4abab = np.zeros([ndocc, ndocc, ndocc, ndocc, nvir, nvir, nvir, nvir])
    
    # Search for the appropriate coefficients using a model Determinant
    
    # Singles: Only the case (alpha -> alpha) is necessary
    
    for i in CAS_holes:
        for a in CAS_particles:
            search = ref.copy()
        
            # anh -> i
            search.anh(i, spin=0)
    
            # cre -> a
            search.cre(a+ndocc, spin=0)
    
            index = determinants.index(search)
            CASdet = determinants[index]
            # The phase guarentees that the determinant in the CI framework is the same as the one created in the CC framework
            CAS_T1[i,a] = Ccas[index] * search.sign() * CASdet.sign()
    
    # Doubles: Only the case (alpha, beta -> alpha, beta) is necessary
    
    for i in CAS_holes:
        for a in CAS_particles:
            for j in CAS_holes:
                for b in CAS_particles:
                    search = ref.copy()
    
                    # anh -> i
                    search.anh(i, spin=0)
    
                    # anh -> j
                    search.anh(j, spin=1)
    
                    # cre -> b
                    search.anh(b + ndocc, spin=1)
    
                    # cre -> a
                    search.cre(a + ndocc, spin=0)
    
                    index = determinants.index(search)
                    CASdet = determinants[index]
                    # The phase guarentees that the determinant in the CI framework is the same as the one created in the CC framework
                    CAS_T2[i,j,a,b] = Ccas[index] * search.sign() * CASdet.sign()
    
    # Triples. Spin case: aba -> aba
    
    for i in CAS_holes:
        for a in CAS_particles:
            for j in CAS_holes:
                for b in CAS_particles:
                    for k in CAS_holes:
                        for c in CAS_particles:
                            # Since ik and ac are alphas, they cannot be the same
                            if i == k or a == c:
                                continue
                            search = ref.copy()
    
                            # anh -> i (alpha)
                            search.anh(i, spin=0)
    
                            # anh -> j (beta)
                            search.anh(j, spin=1)
    
                            # anh -> k (alpha)
                            search.anh(k, spin=0)
    
                            # cre -> c (alpha)
                            search.cre(c + ndocc, spin=0)
    
                            # cre -> b (beta)
                            search.cre(b + ndocc, spin=1)
    
                            # cre -> a (alpha)
                            search.cre(a + ndocc, spin=0)
    
                            index = determinants.index(search)
                            CASdet = determinants[index]
                            CAS_T3aba[i,j,k,a,b,c] = Ccas[index] * search.sign() * CASdet.sign()
    
    # Quadruples 
    
    ## First case: abaa -> abaa
    
    for i in CAS_holes:
        for a in CAS_particles:
            for j in CAS_holes:
                for b in CAS_particles:
                    for k in CAS_holes:
                        for c in CAS_particles:
                            for l in CAS_holes:
                                for d in CAS_particles:
                                    # Same spin indexes cannot be the same
                                    if i == k or i == l or k == l:
                                        continue
                                    if a == c or a == d or c == d:
                                        continue
                                    search = ref.copy()
    
                                    # anh -> i (alpha)
                                    search.anh(i, spin=0)
    
                                    # anh -> j (beta)
                                    search.anh(j, spin=1)
    
                                    # anh -> k (alpha)
                                    search.anh(k, spin=0)
    
                                    # anh -> l (alpha)
                                    search.anh(l, spin=0)
    
                                    # cre -> d (alpha)
                                    search.cre(d + ndocc, spin=0)
    
                                    # cre -> c (alpha)
                                    search.cre(c + ndocc, spin=0)
    
                                    # cre -> b (beta)
                                    search.cre(b + ndocc, spin=1)
    
                                    # cre -> a (alpha)
                                    search.cre(a + ndocc, spin=0)
    
                                    index = determinants.index(search)
                                    CASdet = determinants[index]
                                    CAS_T4abaa[i,j,k,l,a,b,c,d] = Ccas[index] * search.sign() * CASdet.sign()
    
    ## Second case: abab -> abab
    
    for i in CAS_holes:
        for a in CAS_particles:
            for j in CAS_holes:
                for b in CAS_particles:
                    for k in CAS_holes:
                        for c in CAS_particles:
                            for l in CAS_holes:
                                for d in CAS_particles:
                                    # Same spin indexes cannot be the same
                                    if i == k or j == l:
                                        continue
                                    if a == c or b == d:
                                        continue
                                    search = ref.copy()
    
                                    # anh -> i (alpha)
                                    search.anh(i, spin=0)
    
                                    # anh -> j (beta)
                                    search.anh(j, spin=1)
    
                                    # anh -> k (alpha)
                                    search.anh(k, spin=0)
    
                                    # anh -> l (beta)
                                    search.anh(l, spin=1)
    
                                    # cre -> d (beta)
                                    search.cre(d + ndocc, spin=1)
    
                                    # cre -> c (alpha)
                                    search.cre(c + ndocc, spin=0)
    
                                    # cre -> b (beta)
                                    search.cre(b + ndocc, spin=1)
    
                                    # cre -> a (alpha)
                                    search.cre(a + ndocc, spin=0)
    
                                    index = determinants.index(search)
                                    CASdet = determinants[index]
                                    CAS_T4abab[i,j,k,l,a,b,c,d] = Ccas[index] * search.sign() * CASdet.sign()
    
    # Translate CI coefficients into CC amplitudes
    
    print('Translating CI coefficients into CC amplitudes...\n')
    
    # Singles: already done
    
    # Doubles
    CAS_T2 = CAS_T2 - np.einsum('ia,jb-> ijab', CAS_T1, CAS_T1)
    T2aa = CAS_T2 - np.einsum('ijab -> jiab', CAS_T2)
    
    # Triples
    
    CAS_T3aba += - np.einsum('ia,jkbc -> ijkabc', CAS_T1, CAS_T2) \
                      + np.einsum('ic,kjab -> ijkabc', CAS_T1, CAS_T2) \
                      + np.einsum('ka,ijcb -> ijkabc', CAS_T1, CAS_T2) \
                      - np.einsum('kc,ijab -> ijkabc', CAS_T1, CAS_T2) \
                      - np.einsum('jb,ikac -> ijkabc', CAS_T1, T2aa)
    
    CAS_T3aba += - np.einsum('ia,jb,kc -> ijkabc', CAS_T1, CAS_T1, CAS_T1) \
                      + np.einsum('ic,jb,ka -> ijkabc', CAS_T1, CAS_T1, CAS_T1) 
    
    CAS_T3aaa = CAS_T3aba - np.einsum('ijkabc -> ikjabc', CAS_T3aba) - np.einsum('ijkabc -> jikabc', CAS_T3aba)
    print('Translating quadruples...')
    # Quadruples
    
    print('Spin case (ABAA -> ABAA)')
    ## First case abaa -> abaa
    
    CAS_T4abaa = CAS_T4abaa
    
    print('... T1 * T3...')
    ### T1 * T3 terms
    
    CAS_T4abaa += - np.einsum('ia, kjlcbd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('ic, kjlabd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('id, kjlabc -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('jb, iklacd -> ijklabcd', CAS_T1, CAS_T3aaa) \
                       + np.einsum('ka, ijlcbd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('kc, ijlabd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('kd, ijlabc -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('la, ijkcbd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('lc, ijkabd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('ld, ijkabc -> ijklabcd', CAS_T1, CAS_T3aba)
    
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
    
    CAS_T4abab += - np.einsum('ia, jklbcd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('ic, jklbad -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('jb, ilkadc -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('jd, ilkabc -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('ka, jilbcd -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('kc, jilbad -> ijklabcd', CAS_T1, CAS_T3aba) \
                       + np.einsum('lb, ijkadc -> ijklabcd', CAS_T1, CAS_T3aba) \
                       - np.einsum('ld, ijkabc -> ijklabcd', CAS_T1, CAS_T3aba)
    
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

    return CAS_T1, CAS_T2, CAS_T3aba, CAS_T4abab, CAS_T4abaa
