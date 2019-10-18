import numpy as np

temp = '{}{}, {}{}{}{}{}{} -> ijklabcd'

vir  = ['a', 'b', 'c', 'd']
spin = [ 0 ,  1 ,  0 ,  0 ]
oc   = ['i', 'j', 'k', 'l']

strings = []

for s1,v in zip(spin,vir):
    for s2,o in zip(spin,oc):
        if s1 == s2:
            nvir = vir[:]
            nvir.remove(v)
            noc  = oc[:]
            noc.remove(o)
            strings.append(temp.format(o,v,*noc, *nvir))

temp2 = 'np.einsum(\'{}\', self.CAS_T1, self.CAS_T3aaa)'

for i in strings:
    print(temp2.format(i))

        


