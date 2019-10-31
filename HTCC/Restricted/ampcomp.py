def tcompare(t1, t2, t3aaa, t3aba, t4abaa, t4abab, ndocc):

    with open('mrcc_amps.dat') as mrcc_amps:
        values = []
        charc = []
        for lines in mrcc_amps:
            line = lines.split()
            values.append(float(line[0]))
            charc.append(line[1:])
    
    warning = []
    labels = []
    for x in charc:
        newx = ''
        for elements in x:
            newx += elements + ' '
        newx = newx[:-1]
        labels.append(newx)
    
    out = '{:<30s}   {:<15s}   {:<15s}   {:<15s}'.format('Label', 'MRCC', 'HTCC', 'Difference') + '\n'
    
    for q in range(len(values)):
        label = labels[q]
        mrcc_amp = values[q]
    
        # Translate label
    
        particles = []
        holes = []
    
        f = False
        entry = label.split()
        for x in entry:
            if x == '->':
                f = True
                continue
            
            if not f:
                holes.append(x)
            else:
                particles.append(x)
        
        exc_level = len(holes)
    
        # If excitation level is 1 (singles):
    
        if exc_level == 1:
            i = int(holes[0][:-1]) - 1
            i_spin = holes[0][-1]
            
            a = int(particles[0][:-1]) - 1 - ndocc
            a_spin = particles[0][-1]
            if i_spin != a_spin:
                htcc_amp = 0.0
            else:
                htcc_amp = t1[i,a]

        elif exc_level == 2:
            i = int(holes[0][:-1]) - 1
            i_spin = holes[0][-1]

            j = int(holes[1][:-1]) - 1
            j_spin = holes[1][-1]
            
            a = int(particles[0][:-1]) - 1 - ndocc
            a_spin = particles[0][-1]

            b = int(particles[1][:-1]) - 1 - ndocc
            b_spin = particles[1][-1]

            # If all alphas or all betas

            if i_spin == j_spin and b_spin == a_spin:
                htcc_amp = t2[i,j,a,b] - t2[j,i,a,b]

            # If mixed spin:  a b -> a b

            elif i_spin == a_spin and j_spin == b_spin:
                htcc_amp = t2[i,j,a,b]

            # If mixed spin: a b -> b a
            elif i_spin == b_spin and j_spin == i_spin:
                htcc_amp = -t2[j,i,a,b]
            else:
                out += 'UNKNOWN SPIN CASE: {}'.format(label)
                break

        elif exc_level == 3:
            i = int(holes[0][:-1]) - 1
            i_spin = holes[0][-1]

            j = int(holes[1][:-1]) - 1
            j_spin = holes[1][-1]

            k = int(holes[2][:-1]) - 1
            k_spin = holes[2][-1]
            
            a = int(particles[0][:-1]) - 1 - ndocc
            a_spin = particles[0][-1]

            b = int(particles[1][:-1]) - 1 - ndocc
            b_spin = particles[1][-1]

            c = int(particles[2][:-1]) - 1 - ndocc
            c_spin = particles[2][-1]

            # Spin case a a a -> a a a or b b b -> b b b 
            if len(set([i_spin, j_spin, k_spin])) == 1 and len(set([a_spin, b_spin, c_spin])) == 1:
                htcc_amp = t3aaa[i,j,k,a,b,c]

            # Spin case a b b -> a b b 
            elif j_spin == k_spin and b_spin == c_spin:
                htcc_amp = t3aba[j,i,k,b,a,c]

            # Spin case a a b -> a a b 
            elif i_spin == j_spin and a_spin == b_spin:
                htcc_amp = t3aba[i,k,j,a,c,b]
            else:
                out += 'UNKNOWN SPIN CASE: {}'.format(label)
                break
        
        elif exc_level == 4:
            i = int(holes[0][:-1]) - 1
            i_spin = holes[0][-1]

            j = int(holes[1][:-1]) - 1
            j_spin = holes[1][-1]

            k = int(holes[2][:-1]) - 1
            k_spin = holes[2][-1]

            l = int(holes[3][:-1]) - 1
            l_spin = holes[3][-1]
            
            a = int(particles[0][:-1]) - 1 - ndocc
            a_spin = particles[0][-1]

            b = int(particles[1][:-1]) - 1 - ndocc
            b_spin = particles[1][-1]

            c = int(particles[2][:-1]) - 1 - ndocc
            c_spin = particles[2][-1]

            d = int(particles[3][:-1]) - 1 - ndocc
            d_spin = particles[3][-1]

            # Spin case a a b b -> a a b b 
            if len(set([k_spin, l_spin, c_spin, d_spin])) == 1 and len(set([a_spin, b_spin, i_spin, j_spin])) == 1:
                htcc_amp = t4abab[i,k,j,l,a,c,b,d]
            elif len(set([j_spin, k_spin, l_spin, b_spin, c_spin, d_spin])) == 1:
                htcc_amp = t4abaa[j,i,k,l,b,a,c,d]

            else:
                out += 'UNKNOWN SPIN CASE: {}'.format(label)
                break

        out += '{:<30s}   {:< 14.10f}   {:< 14.10f}   {:< 14.10f}'.format(label, mrcc_amp, htcc_amp, abs(mrcc_amp-htcc_amp)) + '\n'

    output = open('tcompare.out', 'w')
    output.write(out)
    output.close()
