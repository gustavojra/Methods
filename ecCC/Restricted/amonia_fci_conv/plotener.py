import numpy as np
import matplotlib.pyplot as plt
import sys

f = True
labels = []
acs = []
CAS = []
TCCSD = []
CASCC = []
acs = ['(4,4)', '(6,6)', '(8,8)', '(10,10)', '(10,11)', '(10,12)', '(10,13)', '(10,14)', '(10,15)']
fci = -56.293108053819665
with open('energies.dat', 'r') as data:
    for line in data:
        if f:
            f = False
            continue
        a, c1, c2, c3 = line.split()
        CAS.append(1000*(float(c1)-fci))
        TCCSD.append(1000*(float(c2)-fci))
        CASCC.append(1000*(float(c3)-fci))
print(acs)
plt.clf()
plt.axhline(y=0, color='black')
plt.plot(acs, TCCSD, 'ro', label='TCCSD')
plt.plot(acs, CASCC, 'bo', label='CASCC')
plt.xlabel('Active Space ($N_e$, $N_{mo}$)',fontsize=14)
plt.ylabel('$E - E_{FCI}$ ($mE_h$)',fontsize=14)
plt.legend()
plt.show()
