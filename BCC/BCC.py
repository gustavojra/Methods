import psi4
import os
import sys
import numpy as np
import time
import copy
import numpy.linalg as npla
import scipy.linalg as sp

np.set_printoptions(suppress=True)

### FUNCTIONS ###


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

def compute_integrals(C, mints):
    h = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    h = np.einsum('up,vq,uv->pq', C, C, h)
    V = np.asarray(mints.mo_eri(C, C, C, C))
    # Convert to physicist notation
    V = V.swapaxes(1,2)
    return h, V

def get_fock(h, V):
    fo = h + 2*np.einsum('piqi->pq', Vint[:,o,:,o]) - np.einsum('piiq->pq', Vint[:,o,o,:])
    fd = np.array(fo.diagonal())
    np.fill_diagonal(fo, 0)
    return fd, fo

def cc_energy(T1, T2):

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1)
    X = 2*tau - np.einsum('ijab->jiab',tau)
    E = np.einsum('abij,ijab->', Vint[v,v,o,o], X)
    return E

def overlap(C, mints):
    S = np.array(mints.ao_overlap())
    Sorb = np.einsum('up,vq,uv->pq', C, C, S)
    return Sorb

def CCSD_Iter(T1, T2, fo, d, D, EINSUMOPT='optimal'):

    # Intermediate arrays

    tau = T2 + np.einsum('ia,jb->ijab', T1, T1,optimize=EINSUMOPT)
    Te = 0.5*T2 + np.einsum('ia,jb->ijab', T1, T1,optimize=EINSUMOPT)

    A2l = np.einsum('uvij,ijpg->uvpg', Vint[o,o,o,o], tau,optimize=EINSUMOPT)
    B2l = np.einsum('abpg,uvab->uvpg', Vint[v,v,v,v], tau,optimize=EINSUMOPT)
    C1  = np.einsum('uaip,ia->uip', Vint[o,v,o,v], T1,optimize=EINSUMOPT) 
    C2  = np.einsum('aupi,viga->pvug', Vint[v,o,v,o], T2,optimize=EINSUMOPT)
    C2l = np.einsum('iaug,ivpa->pvug', Vint[o,v,o,v], tau,optimize=EINSUMOPT)
    D1  = np.einsum('uapi,va->uvpi', Vint[o,v,v,o], T1,optimize=EINSUMOPT)
    D2l = np.einsum('abij,uvab->uvij',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    Ds2l= np.einsum('acij,ijpb->acpb',Vint[v,v,o,o], tau,optimize=EINSUMOPT)
    D2a = np.einsum('baji,vjgb->avig', Vint[v,v,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    D2b = np.einsum('baij,vjgb->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    D2c = np.einsum('baij,vjbg->avig', Vint[v,v,o,o], T2,optimize=EINSUMOPT)
    Es1 = np.einsum('uvpi,ig->uvpg', Vint[o,o,v,o], T1,optimize=EINSUMOPT)
    E1  = np.einsum('uaij,va->uvij', Vint[o,v,o,o], T1,optimize=EINSUMOPT)
    E2a = np.einsum('buji,vjgb->uvig', Vint[v,o,o,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)
    E2b = np.einsum('buij,vjgb->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    E2c = np.einsum('buij,vjbg->uvig', Vint[v,o,o,o], T2,optimize=EINSUMOPT)
    F11 = np.einsum('bapi,va->bvpi', Vint[v,v,v,o], T1,optimize=EINSUMOPT)
    F12 = np.einsum('baip,va->bvip', Vint[v,v,o,v], T1,optimize=EINSUMOPT)
    Fs1 = np.einsum('acpi,ib->acpb', Vint[v,v,v,o], T1,optimize=EINSUMOPT)
    F2a = np.einsum('abpi,uiab->aup', Vint[v,v,v,o], 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT) 
    F2l = np.einsum('abpi,uvab->uvpi', Vint[v,v,v,o], tau,optimize=EINSUMOPT)

    X = E1 + D2l
    giu = np.einsum('ujij->ui', 2*X - X.transpose(0,1,3,2),optimize=EINSUMOPT)
    
    X = Fs1 - Ds2l
    gap = np.einsum('abpb->ap', 2*X - X.transpose(1,0,2,3),optimize=EINSUMOPT)

    # T2 Amplitudes update

    J = np.einsum('ap,uvag->uvpg',fo[v,v],T2) - np.einsum('ui,ivpg->uvpg',fo[o,o],T2) 
    J += np.einsum('ai,uvag,ip->uvpg',fo[v,o],T2,T1) 
    J += np.einsum('ai,uipg,va->uvpg',fo[v,o],T2,T1) 

    J += np.einsum('ag,uvpa->uvpg', gap, T2,optimize=EINSUMOPT) - np.einsum('vi,uipg->uvpg', giu, T2,optimize=EINSUMOPT)

    S = 0.5*A2l + 0.5*B2l - Es1 - (C2 + C2l - D2a - F12).transpose(2,1,0,3)  
    S += np.einsum('avig,uipa->uvpg', (D2a-D2b), T2 - Te.transpose(0,1,3,2),optimize=EINSUMOPT)
    S += 0.5*np.einsum('avig,uipa->uvpg', D2c, T2,optimize=EINSUMOPT)
    S += np.einsum('auig,viap->uvpg', D2c, Te,optimize=EINSUMOPT)
    S += np.einsum('uvij,ijpg->uvpg', 0.5*D2l + E1, tau,optimize=EINSUMOPT)
    S -= np.einsum('uvpi,ig->uvpg', D1 + F2l, T1,optimize=EINSUMOPT)
    S -= np.einsum('uvig,ip->uvpg',E2a - E2b - E2c.transpose(1,0,2,3), T1,optimize=EINSUMOPT)
    S -= np.einsum('avgi,uipa->uvpg', F11, T2,optimize=EINSUMOPT)
    S -= np.einsum('avpi,uiag->uvpg', F11, T2,optimize=EINSUMOPT)
    S += np.einsum('avig,uipa->uvpg', F12, 2*T2 - T2.transpose(0,1,3,2),optimize=EINSUMOPT)

    T2new = Vint[o,o,v,v] + J + J.transpose(1,0,3,2) + S + S.transpose(1,0,3,2)

    T2new = np.einsum('uvpg,uvpg->uvpg', T2new, D,optimize=EINSUMOPT)

    res2 = np.sum(np.abs(T2new - T2))

    # T1 Amplitudes update

    # Fock terms

    T1new = -fo[o,v] + np.einsum('ui,ip->up', fo[o,o], T1) 
    T1new += -np.einsum('ap,ua->up', fo[v,v], T1) + 2*np.einsum('ai,uipa->up',fo[v,o],T2) - np.einsum('ai,uiap->up', fo[v,o], tau)

    # Regular

    T1new += np.einsum('ui,ip->up', giu, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('ap,ua->up', gap, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('juai,ja,ip->up', 2*D1 - D1.transpose(3,1,2,0), T1, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('auip,ia->up', 2*(D2a - D2b) + D2c, T1,optimize=EINSUMOPT)
    T1new -= np.einsum('aup->up', F2a,optimize=EINSUMOPT)
    T1new += np.einsum('uiip->up', 1.0/2.0*(E2a - E2b) + E2c,optimize=EINSUMOPT)
    T1new += np.einsum('uip->up', C1,optimize=EINSUMOPT)
    T1new -= 2*np.einsum('uipi->up', D1,optimize=EINSUMOPT)

    T1new = np.einsum('up,up->up', T1new, d,optimize=EINSUMOPT)
    
    res1 = np.sum(np.abs(T1new - T1))

    return T1new, T2new, res1, res2

# Input Geometry    

#H2 = psi4.geometry("""
#    0 1
#    H 
#    H 1 0.76
#    symmetry c1
#""")

water = psi4.geometry("""
    O
    H 1 R
    H 1 R 2 A
    
    R = .9
    A = 104.5
    symmetry c1
""")

#ethane = psi4.geometry("""
#    0 1
#    C       -3.4240009952      1.7825072183      0.0000001072                 
#    C       -1.9048206760      1.7825072100     -0.0000000703                 
#    H       -3.8005812586      0.9031676785      0.5638263076                 
#    H       -3.8005814434      1.7338892156     -1.0434433083                 
#    H       -3.8005812617      2.7104647651      0.4796174543                 
#    H       -1.5282404125      0.8545496587     -0.4796174110                 
#    H       -1.5282402277      1.8311252186      1.0434433449                 
#    H       -1.5282404094      2.6618467448     -0.5638262767  
#    symmetry c1
#""")

#form = psi4.geometry("""
#0 1
#O
#C 1 1.22
#H 2 1.08 1 120.0
#H 2 1.08 1 120.0 3 -180.0
#symmetry c1
#""")

# Basis set

basis = 'sto-3g'

# Psi4 Options

psi4.core.be_quiet()
psi4.set_options({'basis': basis,
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence' : 1e-10,
                  'freeze_core': 'false'})

# Run Psi4 Energie
print('---------------- RUNNING PSI4 ------------------')
tinit = time.time()
scf_e, wfn = psi4.energy('scf', return_wfn=True)
#p4_ccsd = psi4.energy('ccsd')
#p4_bccd = psi4.energy('bccd')
#
#print('SCF  Energy from Psi4: {:<5.10f}'.format(scf_e))
#print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_ccsd))
#print('------------------------------------------------')
#print('Psi4 computations completed in {:.5f} seconds\n'.format(time.time() - tinit))

print('Collecting Hartree-Fock orbitals\n')
nelec = wfn.nalpha() + wfn.nbeta()
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
nvir = nmo - ndocc
fd = np.asarray(wfn.epsilon_a())
nbf = C.shape[0]

print("Number of Basis Functions:      {}".format(nbf))
print("Number of Electrons:            {}".format(nelec))
print("Number of Molecular Orbitals:   {}".format(nmo))
print("Number of Doubly ocuppied MOs:  {}".format(ndocc))

# Get Integrals

print("Converting atomic integrals to MO integrals...")
mints = psi4.core.MintsHelper(wfn.basisset())
h, Vint = compute_integrals(C, mints)

# Slices

o = slice(0, ndocc)
v = slice(ndocc, nbf)

# Compute Fock Matrix

fd, fo = get_fock(h, Vint)

# START CCSD CODE

# Build the Auxiliar Matrix D

print('\n----------------- RUNNING CCD ------------------')

print('\nBuilding Auxiliar D matrix...')
t = time.time()
D  = np.zeros([ndocc, ndocc, nvir, nvir])
d  = np.zeros([ndocc, nvir])

for i,ei in enumerate(fd[o]):
    for j,ej in enumerate(fd[o]):
        for a,ea in enumerate(fd[v]):
            d[i,a] = 1/(ea - ei)
            for b,eb in enumerate(fd[v]):
                D[i,j,a,b] = 1/(ei + ej - ea - eb)

print('\nComputing MP2 guess')

t = time.time()

T1 = np.zeros([ndocc, nvir])
T2 = np.einsum('abij,ijab->ijab', Vint[v,v,o,o], D)

E = cc_energy(T1, T2)

print('MP2 Cor Energy: {:<5.10f}     Time required: {:.5f}'.format(E, time.time()-t))
print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(E+scf_e, time.time()-t))

r1 = 0
r2 = 1
CC_CONV = 8
CC_MAXITER = 30
    
LIM = 10**(-CC_CONV)

ite = 0

while r2 > LIM or r1 > LIM:
    ite += 1
    if ite > CC_MAXITER:
        raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
    Eold = E
    t = time.time()
    T1, T2, r1, r2 = CCSD_Iter(T1, T2, fo, d, D)
    E = cc_energy(T1, T2)
    dE = E - Eold
    print('-'*50)
    print("Iteration {}".format(ite))
    print("CC Correlation energy: {}".format(E))
    print("Energy change:         {}".format(dE))
    print("T1 Residue:            {}".format(r1))
    print("T2 Residue:            {}".format(r2))
    print("Max T1 Amplitude:      {}".format(np.max(T1)))
    print("Max T2 Amplitude:      {}".format(np.max(T2)))
    print("Time required:         {}".format(time.time() - t))
    print('-'*50)

print("\nCC Equations Converged!!!")
print("Final CCSD Energy:     {:<5.10f}".format(E + scf_e))
print('CCSD Energy from Psi4: {:<5.10f}'.format(psi4.energy('ccsd')))
#print('CCSD Energy from Psi4: {:<5.10f}'.format(p4_bccd))

newC = C.to_array()

while T1.max() > 1e-8:

    X = np.block([
                   [ np.zeros([ndocc, ndocc]), np.zeros_like(T1)],
                   [ T1.T, np.zeros([nvir, nvir])]])

    print('Via Tai')
    U = sp.expm(X - X.T)
    newC = newC.dot(U)

#    print('Rotating Orbitals...')
    
    oldC = copy.deepcopy(newC)
#
#    def rotate(i,a):
#        T = T1[i,a]
#        a = a + ndocc
#        newC[:,i] = newC[:,i] + oldC[:,a]*T
#        newC[:,a] = newC[:,a] - oldC[:,i]*T
#
#    print('OLD')
#    Sao = np.array(mints.ao_overlap())
#    printmatrix(oldC)
#    #S = overlap(psi4.core.Matrix.from_array(newC), mints)
#    
#    printmatrix(oldC.T.dot(Sao).dot(oldC))
#    
#    for i in range(ndocc):
#        for a in range(nvir):
#            rotate(i,a)
#            
#    print('New')
#    printmatrix(newC)
#    S = overlap(psi4.core.Matrix.from_array(newC), mints)
#    
#    printmatrix(S)
#
#    newC = np.zeros([nbf, nbf])
#
#    newC[:,o] = oldC[:,o] + np.einsum('ua,ia->ui', oldC[:,v], T1)
#    newC[:,v] = oldC[:,v] - np.einsum('ui,ia->ua', oldC[:,o], T1)
#
#    print('Via einsum')
#    printmatrix(newC)
#
#    raise NameError('Stop')
    
#
#    S = overlap(psi4.core.Matrix.from_array(newC), mints)
#    
#    printmatrix(S)

    
    h, Vint = compute_integrals(psi4.core.Matrix.from_array(newC), mints)
    
    fd, fo = get_fock(h, Vint)
    
    print('Run CC again...')
    D  = np.zeros([ndocc, ndocc, nvir, nvir])
    d  = np.zeros([ndocc, nvir])
    
    for i,ei in enumerate(fd[o]):
        for j,ej in enumerate(fd[o]):
            for a,ea in enumerate(fd[v]):
                d[i,a] = 1/(ea - ei)
                for b,eb in enumerate(fd[v]):
                    D[i,j,a,b] = 1/(ei + ej - ea - eb)
    
    print('\nComputing MP2 guess')
    
    t = time.time()
    
    T1 = np.zeros([ndocc, nvir])
    T2 = np.einsum('abij,ijab->ijab', Vint[v,v,o,o], D)
    
    E = cc_energy(T1, T2)
    
    print('MP2 Cor Energy: {:<5.10f}     Time required: {:.5f}'.format(E, time.time()-t))
    print('MP2 Energy: {:<5.10f}     Time required: {:.5f}'.format(E+scf_e, time.time()-t))
    
    r1 = 0
    r2 = 1
    CC_CONV = 8
    CC_MAXITER = 30
        
    LIM = 10**(-CC_CONV)
    
    ite = 0
    
    while r2 > LIM or r1 > LIM:
        ite += 1
        if ite > CC_MAXITER:
            raise NameError("CC Equations did not converge in {} iterations".format(CC_MAXITER))
        Eold = E
        t = time.time()
        T1, T2, r1, r2 = CCSD_Iter(T1, T2, fo, d, D)
        E = cc_energy(T1, T2)
        dE = E - Eold
        print('-'*50)
        print("Iteration {}".format(ite))
        print("CC Correlation energy: {}".format(E))
        print("Energy change:         {}".format(dE))
        print("T1 Residue:            {}".format(r1))
        print("T2 Residue:            {}".format(r2))
        print("Max T1 Amplitude:      {}".format(np.max(T1)))
        print("Max T2 Amplitude:      {}".format(np.max(T2)))
        print("Time required:         {}".format(time.time() - t))
        print('-'*50)
    
    print("\nCC Equations Converged!!!")
    print("Final CCSD Energy:     {:<5.10f}".format(E + scf_e))

print('FInal overlap')
S = overlap(psi4.core.Matrix.from_array(newC), mints)
printmatrix(S)
eccsd = psi4.energy('ccsd')
ebccd = psi4.energy('bccd')
print('Psi4 CCSD {:<5.10f}'.format(eccsd))
print('Psi4 BCCD {:<5.10f}'.format(ebccd))