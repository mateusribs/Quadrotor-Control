import numpy as np
import sympy as sym
from sympy.matrices.expressions.blockmatrix import BlockMatrix



def polyT(n, k, t):

    T = np.zeros((n,1))
    D = np.zeros((n,1))

    for i in range(0, n-1):
        D[i] = i - 1
        T[i] = 1
    
    for j in range(0, k-1):
        for i in range(0, n-1):
            T[i] = T[i]*D[i]

            if D[i]>0:
                D[i] = D[i] - 1
            

    for i in range(0, n-1):
        T[i] = T[i]*t**D[i]
    
    T = T.T

    return T


T = polyT(8, 4, 1)
print(T)