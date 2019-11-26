import numpy as np
import scipy.linalg as la
import time
from davidson import Davidson
import matplotlib.pyplot as plt

def compare(n):
    sparsity = 0.00001
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = i + 1
    A = A + sparsity*np.random.randn(n,n)
    A = (A.T + A)/2
    
    t = time.time()
    E, C = la.eigh(A)
    tla = time.time() - t
    
    t = time.time()
    E, C = Davidson(A)
    tdav = time.time() - t
    
    return tla, tdav

sizes = list(range(100,5000,100))
tla = []
tdav = []
print(sizes)

for n in sizes:
    print('Running n = {}'.format(n))
    x,y = compare(n)
    tla.append(x)
    tdav.append(y)
print(tla)

plt.plot(sizes, tla, 'r', label='numpy linalg')
plt.plot(sizes, tdav, 'b', label='Davidson')
plt.ylabel('Time (s)')
plt.xlabel('Matrix size')
plt.show()

 
