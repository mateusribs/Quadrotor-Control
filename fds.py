import numpy as np
import sympy as sym
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt




def polyT(n, k, t):

    T = np.zeros((n,1))
    D = np.zeros((n,1))

    for i in range(1, n+1):
        D[i-1] = i - 1
        T[i-1] = 1

    for j in range(1, k+1):
        for i in range(1, n+1):
            T[i-1] = T[i-1]*D[i-1]

            if D[i-1]>0:
                D[i-1] = D[i-1] - 1
            

    for i in range(1, n+1):
        T[i-1] = T[i-1]*t**D[i-1]
    
    T = T.T

    return T

def getCoeff_snap(waypoints, t):

    n = len(waypoints) - 1
    A = np.zeros((8*n, 8*n))
    b = np.zeros((8*n, 1))

    # print(b.T)

    row = 0
    #Initial constraints
    for i in range(0, 1):
        A[row, 8*(i):8*(i+1)] = polyT(8, 0, t[0])
        b[i, 0] = waypoints[0]
        row = row + 1
    
    for k in range(1, 4):
        A[row, 0:8] = polyT(8, k, t[0])
        row = row + 1
    
    if n == 1:
        #Last P constraints
        for i in range(0, 1):
            A[row, 8*(i):8*(i+1)] = polyT(8, 0, t[-1])
            b[row, 0] = waypoints[1]
            row = row + 1  
        
        for k in range(1, 4):
            A[row, 8*(n) - 8:8*(n)] = polyT(8, k, t[-1])
            row = row + 1



    elif n>1:


        #Pi constraints
        shift = 0
        for j in range(1, n):
            
            
            
            for i in range(0, 2):
                A[row, 8*(i+shift):8*(i+1+shift)] = polyT(8, 0, t[j])
                b[row, 0] = waypoints[j]

                row = row + 1

            for k in range(1, 7):
                A[row, 8*(j-1):8*(j)] = polyT(8, k, t[j])
                A[row, 8*(j):8*(j+1)] = -polyT(8, k, t[j])
                row = row + 1
            
            shift += 1
        
        
        #Last P constraints
        for i in range(0, 1):
            A[row, 8*(n) - 8:8*(n)] = polyT(8, 0, t[-1])
            b[row, 0] = waypoints[n]
            row = row + 1
        
        for k in range(1, 4):
            A[row, 8*(n) - 8:8*(n)] = polyT(8, k, t[-1])
            row = row + 1


    coeff = np.linalg.inv(A)@b
    
    c_matrix = coeff.reshape(n, 8)

    return A, b, c_matrix

def equation_snap(t, c_matrix, eq_n):
    x = polyT(8, 0, t)
    v = polyT(8, 1, t)
    a = polyT(8, 2, t)
    j = polyT(8, 3, t)
    s = polyT(8, 4, t)

    P = np.sum(x*c_matrix[eq_n,:])
    V = np.sum(v*c_matrix[eq_n,:])
    A = np.sum(a*c_matrix[eq_n,:])
    J = np.sum(j*c_matrix[eq_n,:])
    S = np.sum(s*c_matrix[eq_n,:])
    

    return P, V, A, J, S
    
def evaluate_equations_snap(t, c_matrix):
        
    skip = 0

    x_list = []
    v_list = []
    a_list = []
    j_list = []
    s_list = []

    for i in np.arange(0, t[-1], 0.01):

        if skip == 0:

            if i >= t[skip] and i<=t[skip+1]:
            
                p, v, a, j, s = equation_snap(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                s_list.append(s)

            else:

                skip += 1

                p, v, a, j, s = equation_snap(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                s_list.append(s)

        elif skip > 0 and skip < len(t):

            if i > t[skip] and i <= t[skip+1]:

                p, v, a, j, s = equation_snap(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                s_list.append(s)

            else:

                skip += 1

                p, v, a, j, s = equation_snap(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)
                j_list.append(j)
                s_list.append(s)
    
    return x_list, v_list, a_list, j_list, s_list



def getCoeff_accel(waypoints, t):

    n = len(waypoints) - 1
    A = np.zeros((4*n, 4*n))
    b = np.zeros((4*n, 1))

    # print(b.T)

    row = 0
    #Initial constraints
    for i in range(0, 1):
        A[row, 4*(i):4*(i+1)] = polyT(4, 0, t[0])
        b[i, 0] = waypoints[0]
        row = row + 1
    
    for k in range(1, 2):
        A[row, 0:4] = polyT(4, k, t[0])
        row = row + 1
    
    if n == 1:
        #Last P constraints
        for i in range(0, 1):
            A[row, 4*(i):4*(i+1)] = polyT(4, 0, t[-1])
            b[row, 0] = waypoints[1]
            row = row + 1  
        
        for k in range(1, 2):
            A[row, 4*(n) - 4:4*(n)] = polyT(4, k, t[-1])
            row = row + 1



    elif n>1:


        #Pi constraints
        shift = 0
        for j in range(1, n):
            
            
            
            for i in range(0, 2):
                A[row, 4*(i+shift):4*(i+1+shift)] = polyT(4, 0, t[j])
                b[row, 0] = waypoints[j]

                row = row + 1

            for k in range(1, 3):
                A[row, 4*(j-1):4*(j)] = polyT(4, k, t[j])
                A[row, 4*(j):4*(j+1)] = -polyT(4, k, t[j])
                row = row + 1
            
            shift += 1
        
        
        #Last P constraints
        for i in range(0, 1):
            A[row, 4*(n) - 4:4*(n)] = polyT(4, 0, t[-1])
            b[row, 0] = waypoints[n]
            row = row + 1
        
        for k in range(1, 2):
            A[row, 4*(n) - 4:4*(n)] = polyT(4, k, t[-1])
            row = row + 1


    coeff = np.linalg.inv(A)@b
    
    c_matrix = coeff.reshape(n, 4)

    return A, b, c_matrix

def equation_accel(t, c_matrix, eq_n):
    x = polyT(4, 0, t)
    v = polyT(4, 1, t)
    a = polyT(4, 2, t)

    P = np.sum(x*c_matrix[eq_n,:])
    V = np.sum(v*c_matrix[eq_n,:])
    A = np.sum(a*c_matrix[eq_n,:])
    

    return P, V, A

def evaluate_equations_accel(t, c_matrix):
        
    skip = 0

    x_list = []
    v_list = []
    a_list = []

    for i in np.arange(0, t[-1], 0.01):

        if skip == 0:

            if i >= t[skip] and i<=t[skip+1]:
            
                p, v, a = equation_accel(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)

            else:

                skip += 1

                p, v, a = equation_accel(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)

        elif skip > 0 and skip < len(t):

            if i > t[skip] and i <= t[skip+1]:

                p, v, a = equation_accel(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)

            else:

                skip += 1

                p, v, a = equation_accel(i, c_matrix, skip)

                x_list.append(p)
                v_list.append(v)
                a_list.append(a)
    
    return x_list, v_list, a_list


x_wp = np.array([[0, 1, 1, 1, 2]]).T
y_wp = np.array([[0, 0.5, 1, 1.5, 2.5]]).T
z_wp = np.array([[0, 1, 2, 3, 4]]).T
psi_wp = np.array([[0, np.pi, 2*np.pi]]).T

t = [0, 1, 2, 3, 4]

_, _, x_matrix = getCoeff_snap(x_wp, t)
_, _, y_matrix = getCoeff_snap(y_wp, t)
_, _, z_matrix = getCoeff_snap(z_wp, t)

px, vx, ax, jx, sx = evaluate_equations_snap(t, x_matrix)
py, vy, ay, jy, sy = evaluate_equations_snap(t, y_matrix)
pz, vz, az, jz, sz = evaluate_equations_snap(t, z_matrix)

# _, _, psi_matrix = getCoeff_accel(psi_wp, t)
# psi, vpsi, apsi = evaluate_equations_accel(t, psi_matrix)

# t1 = np.arange(0, t[-1], 0.01)

# fig, (p, v, a) = plt.subplots(3, 1, figsize=(9,9))
# p.plot(t1, psi)
# v.plot(t1, vpsi)
# a.plot(t1, apsi)

# fig, (p, v, a, j, s) = plt.subplots(5,1, figsize=(9,9))
# p.plot(t1, px)
# v.plot(t1, vx)
# a.plot(t1, ax)
# j.plot(t1, jx)
# s.plot(t1, sx)
# p.grid()
# v.grid()
# a.grid()
# j.grid()
# s.grid()

# fig2, (p2, v2, a2, j2, s2) = plt.subplots(5,1, figsize=(9,9))
# p2.plot(t1, py)
# v2.plot(t1, vy)
# a2.plot(t1, ay)
# j2.plot(t1, jy)
# s2.plot(t1, sy)
# p2.grid()
# v2.grid()
# a2.grid()
# j2.grid()
# s2.grid()

# fig3, (p3, v3, a3, j3, s3) = plt.subplots(5,1, figsize=(9,9))
# p3.plot(t1, pz)
# v3.plot(t1, vz)
# a3.plot(t1, az)
# j3.plot(t1, jz)
# s3.plot(t1, sz)
# p3.grid()
# v3.grid()
# a3.grid()
# j3.grid()
# s3.grid()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(px, py, pz, 'green')

# plt.show()