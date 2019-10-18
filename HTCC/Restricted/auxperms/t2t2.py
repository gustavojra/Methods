temp = 'np.einsum(\'{}j{}b, {}{}{}{} -> ijklabcd\', self.CAS_T2, T2aa)'


vir  = ['a', 'c', 'd']
oc   = ['i', 'k', 'l']


for x in vir:
    for y in oc:
        nvir = vir[:]
        nvir.remove(x)
        noc = oc[:]
        noc.remove(y)
        print(temp.format(y, x, *noc, *nvir))
