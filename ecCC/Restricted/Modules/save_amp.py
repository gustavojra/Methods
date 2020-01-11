import numpy as np

def write_T1(T1):
    (h,p) = T1.shape
    out = ''
    for i in range(h):
        for a in range(p):
            out += str(T1[i,a])
            out += '\n'

    with open('T1.dat', 'w') as t1:
        t1.write(out)

def write_T2(T2):
    (h,h,p,p) = T2.shape
    out = ''
    for i in range(h):
        for j in range(h):
            for a in range(p):
                for b in range(p):
                    out += str(T2[i,j,a,b])
                    out += '\n'

    with open('T2.dat', 'w') as t2:
        t2.write(out)

def read_T1(h, p, t1 = 'T1.dat'):
    T1 = np.zeros([h,p])
    with open(t1, 'r') as t1:
        amps = t1.readlines()
        count = 0
        for i in range(h):
            for a in range(p):
                T1[i,a] = float(amps[count])
                count += 1
    return T1
    
def read_T2(h, p, t2 = 'T2.dat'):
    T2 = np.zeros([h,h,p,p])
    with open(t2, 'r') as t2:
        amps = t2.readlines()
        count = 0
        for i in range(h):
            for j in range(h):
                for a in range(p):
                    for b in range(p):
                        T2[i,j,a,b] = float(amps[count])
                        count += 1
    return T2


