import re

patt_cas = '@@@ Final CASCI\s+?energy:\s*?(.\d*?\.\d*)'
patt_tcc = '@@@ Final TCCSD\s+?energy:\s*?(.\d*?\.\d*)'
patt_cascc = '@@@ Final CASCCSD\s+?energy:\s*?(.\d*?\.\d*)'
c = re.compile(patt_cas)
y = re.compile(patt_tcc)
x = re.compile(patt_cascc)
with open('output.dat') as out:
    out_str = out.read()
    print(c.search(out_str).group(1))
    print(x.search(out_str).group(1))
    print(y.search(out_str).group(1))
