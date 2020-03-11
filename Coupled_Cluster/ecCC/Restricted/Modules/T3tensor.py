import numpy as np

class T3tensor:

    # Create a tensor T(ijk,abc) where i > j and a > c
    # Specifically for a spin case T(aba,aba)

    def __init__(self, h, p):

        self.array = []

        # Creates an empty array from hole and particle sizes
        for i in range(h):
            hold = []
            for a in range(p):
                if i == 0 or a == 0:
                    hold.append(0)
                else:
                    hold.append(np.zeros((h,h-1,p,p-1)))
            self.array.append(hold)

    def __getitem__(self, key):
    
        if len(key) != 6:
            raise KeyError('Key for T3tensor must be 6 integers')

        i,j,k,a,b,c = key

        if i > k and a > c:
            return self.array[i][a][j,k,b,c]
        elif i > k and a < c:
            return self.array[i][c][j,k,b,a]
        elif i < k and a > c:
            return self.array[k][a][j,i,b,c]
        elif i < k and a < c:
            return self.array[k][c][j,i,b,a]
        else:
            return 0
    
    def __setitem__(self, key, value):
        if len(key) != 6:
            raise KeyError('Key for T3tensor must be 6 integers')

        i,j,k,a,b,c = key

        if i > k and a > c:
            self.array[i][a][j,k,b,c] = value
        elif i > k and a < c:
            self.array[i][c][j,k,b,a] = value
        elif i < k and a > c:
            self.array[k][a][j,i,b,c] = value
        elif i < k and a < c:
            self.array[k][c][j,i,b,a] = value
        else:
            return 0




x = T3tensor(2,4)
x[1,1,0,3,2,1] = 7
print(x[1,1,0,1,2,3])
print(x[0,1,1,1,2,3])
