import sys

# This package contains auxiliar functions, useful for debugging and such

# Emojis. Very important stuff

def emoji(key):
    stored = {
    "viva"   : b'\xF0\x9F\x8E\x89'.decode('utf-8'),
    "eyes"   : b'\xF0\x9F\x91\x80'.decode('utf-8'),
    "cycle"  : b'\xF0\x9F\x94\x83'.decode('utf-8'),
    "crying" : b'\xF0\x9F\x98\xAD'.decode('utf-8'),
    "pleft"  : b'\xF0\x9F\x91\x88'.decode('utf-8'),
    "whale"  : b'\xF0\x9F\x90\xB3'.decode('utf-8')
    }
    return stored[key]

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
            
    



