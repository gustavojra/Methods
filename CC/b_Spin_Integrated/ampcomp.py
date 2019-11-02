def tcompare(t1, t2, ndocc):

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
    
    out = '{:<30s}   {:<15s}   {:<15s}   {:<15s}   {}'.format('Label', 'MRCC', 'CC', 'Difference', 'Sign') + '\n'
    
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
        if mrcc_amp/abs(mrcc_amp) == htcc_amp/abs(htcc_amp):
            sign = ''
        else:
            sign = 'INVERTED'
        out += '{:<30s}   {:< 14.10f}   {:< 14.10f}   {:< 14.10f}   {}'.format(label, mrcc_amp, htcc_amp, abs(abs(mrcc_amp)-abs(htcc_amp)), sign) + '\n'

    output = open('tcompare.out', 'w')
    output.write(out)
    output.close()
