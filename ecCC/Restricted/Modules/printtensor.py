import psi4
import numpy as np

def printtensor(T):
    dims = np.shape(T)
    rank = len(dims)
    if rank == 4:
        i_range = range(0,dims[0])
        j_range = range(0,dims[1])
        for i in i_range:
            for j in j_range:
                psi4.core.print_out('Page ({},{},*,*)\n'.format(i,j))
                printmatrix(T[i,j,:,:])
    if rank == 3:
        i_range = range(0,dims[0])
        for i in i_range:
            psi4_core.print_out('Page ({},*,*)\n'.format(i))
            printmatrix(T[i,:,:])

    if rank == 2:
        printmatrix(T)

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
            out += '{:< 9.7f}    '.format(M[i,j])
        out += '\n'
    psi4.core.print_out(out)
