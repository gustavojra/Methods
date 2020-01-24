import sys
import numpy as np

# This package contains auxiliar functions, useful for debugging and such

# Emojis. Very important stuff

def emoji(key):
    stored = {
    "viva"   : b'\xF0\x9F\x8E\x89'.decode('utf-8'),
    "eyes"   : b'\xF0\x9F\x91\x80'.decode('utf-8'),
    "cycle"  : b'\xF0\x9F\x94\x83'.decode('utf-8'),
    "bolt"   : b'\xE2\x9A\xA1'.decode('utf-8'),
    "pin"    : b'\xF0\x9F\x93\x8C'.decode('utf-8'),
    "crying" : b'\xF0\x9F\x98\xAD'.decode('utf-8'),
    "pleft"  : b'\xF0\x9F\x91\x88'.decode('utf-8'),
    "whale"  : b'\xF0\x9F\x90\xB3'.decode('utf-8'),
    "books"  : b'\xF0\x9F\x93\x9A'.decode('utf-8'),
    "check"  : b'\xE2\x9C\x85'.decode('utf-8'),
    "0"      : b'\x30\xE2\x83\xA3'.decode('utf-8'),
    "1"      : b'\x31\xE2\x83\xA3'.decode('utf-8'),
    "2"      : b'\x32\xE2\x83\xA3'.decode('utf-8'),
    "3"      : b'\x33\xE2\x83\xA3'.decode('utf-8'),
    "4"      : b'\x34\xE2\x83\xA3'.decode('utf-8'),
    "5"      : b'\x35\xE2\x83\xA3'.decode('utf-8'),
    "6"      : b'\x36\xE2\x83\xA3'.decode('utf-8'),
    "7"      : b'\x37\xE2\x83\xA3'.decode('utf-8'),
    "8"      : b'\x38\xE2\x83\xA3'.decode('utf-8'),
    "9"      : b'\x39\xE2\x83\xA3'.decode('utf-8'),
    }
    return stored[key]

def numoji(i):
    i = str(i)
    out = ''
    for l in i:
        out += emoji(l) + ' '
    return out

# Clean up numerical zeros

def chop(number):
    if abs(number) < 1e-12:
        return 0
    else:
        return number

# Print a pretty matrix

def pretty(inp):
    try:
        Mat = inp.tolist()
    except:
        Mat = inp
    out = ''
    for row in Mat:
        for x in row:
            out += ' {:>12.7f}'.format(chop(x))
        out += '\n'
    return out

# Progress bar  
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def showout(i, total, size, prefix, file):
    x = int(size*i/total)
    file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), i, total))
    file.flush()
    
def compare_amps(t1, t2ab, t3aaa, t3aba, t4abaa, t4abab):
    with open('mrcc_amps.dat') as mrcc_amps:
        values = []
        labels = []
        for lines in mrcc_amps:
            line = lines.split(' ', 1)
            values.append(float(line[0]))
            labels = line[1:]
        print(labels)

def printcast1(t1, nvir, ndocc, w=False):
    out = ''
    for i in range(ndocc):
        for a in range(nvir):
            x = t1[i,a]
            if abs(x) > 1e-10:
                out += '{}a -> {}a {:< 5.8e}'.format(i+1,1+a+ndocc,x)
                out += '\n'
    if w:
        fil = open('t1amps.dat','w') 
        fil.write(out)
        fil.close()
    else:
        print(out)

def printcast2(t2, nvir, ndocc, w=False):
    out = ''
    for i in range(ndocc):
        for j in range(ndocc):
            if j > i: break
            for a in range(nvir):
                for b in range(nvir):
                    if b > a: break
                    x = t2[i,j,a,b]
                    if abs(x) > 1e-10:
                        out += '{}a {}b -> {}a {}b {:< 5.8e}'.format(i+1,j+1,1+a+ndocc,1+b+ndocc,x)
                        out += '\n'
    if w:
        fil = open('t2amps.dat','w') 
        fil.write(out)
        fil.close()
    else:
        print(out)

def printcast3(t3aba, t3aaa, ndocc, nvir, w=False):
    out = ''
    for i in range(ndocc):
        for j in range(ndocc):
            if j > i: break
            for k in range(ndocc):
                if k > j: break
                for a in range(nvir):
                    for b in range(nvir):
                        if b > a: break
                        for c in range(nvir):
                            if c > b: break
                            x = t3aba[i,j,k,a,b,c]
                            if abs(x) > 1e-10:
                                out += '{}a {}b {}a -> {}a {}b {}a  {:< 5.8e}'.format(i+1,j+1,k+1,1+a+ndocc,1+b+ndocc,1+c+ndocc,x)
                                out += '\n'
                            y = t3aaa[i,j,k,a,b,c]
                            if abs(y) > 1e-10:
                                out += '{}a {}a {}a -> {}a {}a {}a  {:< 5.8e}'.format(i+1,j+1,k+1,1+a+ndocc,1+b+ndocc,1+c+ndocc,y)
                                out += '\n'
    if w:
        fil = open('t3amps.dat','w') 
        fil.write(out)
        fil.close()
    else:
        print(out)

def printcast4(t4abaa, t4abab, ndocc, nvir, w=False):
    out = ''
    for i in range(ndocc):
        for j in range(ndocc):
            if j > i: break
            for k in range(ndocc):
                if k > j: break
                for a in range(nvir):
                    for b in range(nvir):
                        if b > a: break
                        for c in range(nvir):
                            if c > b: break
                            for d in range(nvir):
                                if d > c: break
                                for l in range(ndocc):
                                    if l > k: break
                                    x = t4abaa[i,j,k,l,a,b,c,d]
                                    if abs(x) > 1e-10:
                                        out += '{}a {}b {}a {}a -> {}a {}b {}a {}a {:< 5.8e}'.format(i+1,j+1,k+1,l+1,1+a+ndocc,1+b+ndocc,1+c+ndocc,1+d+ndocc,x)
                                        out += '\n'
                                    x = t4abab[i,j,k,l,a,b,c,d]
                                    if abs(x) > 1e-6:
                                        out += '{}a {}b {}a {}b -> {}a {}b {}a {}b {:< 5.8e}'.format(i+1,j+1,k+1,l+1,1+a+ndocc,1+b+ndocc,1+c+ndocc,1+d+ndocc,x)
                                        out += '\n'
    if w:
        fil = open('t4amps.dat','w') 
        fil.write(out)
        fil.close()
    else:
        print(out)

def printtensor(T):
    dims = np.shape(T)
    rank = len(dims)
    i_range = range(0,dims[0])
    j_range = range(0,dims[1])
    for i in i_range:
        for j in j_range:
            print('Page ({},{},*,*)\n'.format(i,j))
            printmatrix(T[i,j,:,:])

def printmatrix(M):
    dims = np.shape(M)
    rank = len(dims)
    out = ' '*5
    i_range = range(0,dims[0])
    j_range = range(0,dims[1])
    # header
    for j in j_range:
        out += '{:>11}    '.format(j)
    out += '\n'
    for i in i_range:
        out += '{:3d}  '.format(i)
        for j in j_range:
            out += '{:< 10.8f}    '.format(M[i,j])
        out += '\n'
    print(out)
