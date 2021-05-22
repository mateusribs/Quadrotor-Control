import numpy as np
import sympy as sym
from sympy.matrices.expressions.blockmatrix import BlockMatrix



m, g, omega, Ixx, Iyy, Izz, Ir = sym.symbols('m, g, omega, Ixx, Iyy, Izz, Ir', real=True)


A_a = sym.Matrix([[0, 0, 0, 0.5, 0, 0],
                  [0, 0, 0, 0, 0.5, 0],
                  [0, 0, 0, 0, 0, 0.5],
                  [0, 0, 0, 0, Ir*omega/Ixx, 0],
                  [0, 0, 0, -Ir*omega/Iyy, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

B_a = sym.Matrix([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1/Ixx, 0, 0],
                  [0, 1/Iyy, 0],
                  [0, 0, 1/Izz]])


Mc = sym.Matrix(BlockMatrix([B_a, A_a*B_a, A_a**2*B_a, A_a**3*B_a, A_a**4*B_a, A_a**5*B_a]))
print(Mc.rank())



A_t = sym.Matrix([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

B_t = sym.Matrix([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [g, 0, 0],
                  [0, -g, 0],
                  [0, 0, 1/m]])

Mc_2 = sym.Matrix(BlockMatrix([B_t, A_t*B_t, A_t**2*B_t, A_t**3*B_t, A_t**4*B_t, A_t**5*B_t]))
print(Mc_2.rank())