from quadrotor_env import quad
from quadrotor_control import Controller
from quaternion_euler_utility import euler_quat
from scipy.linalg import solve_continuous_are as solve_lqr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Quadrotor Model Object
quad_model = quad(t_step=0.001, n=10, training=False, euler=0, direct_control=0, T=1, clipped=True)
# seed = quad_model.seed(2)

#Constants
J = np.array([[16.83*10**-3, 0, 0],
            [0, 16.83*10**-3, 0],
            [0, 0, 28.34*10**-3]])
#Mass and Gravity
M, G = 1.03, 9.82

KT = 1.435*10**-5
KD = 2.4086*10**-7
L = 0.26

#Trajectory Generator ---> Obtain positions, velocities and accelerations references vectors
inner_length = 50
controller = Controller(total_time = 10, sample_time = 0.001, inner_length = inner_length)
x_ref, dotx_ref, ddotx_ref, y_ref, doty_ref, ddoty_ref, z_ref, dotz_ref, ddotz_ref, psiInt = controller.trajectory_generator(radius=1, frequency=np.pi/6, max_h=4, min_h=4)
outer_length = len(x_ref)


#Linearized System - Quaternion Approach

zeros = np.zeros((3,3))
#Attitude Dynamics

A12_a = np.eye(3)*0.5

A_up_a = np.concatenate((zeros, A12_a), axis=1)
A_down_a = np.concatenate((zeros, zeros), axis=1)
A_a = np.concatenate((A_up_a, A_down_a), axis=0)

B21_a = np.array([[1/J[0,0], 0, 0], [0, 1/J[1,1], 0], [0, 0, 1/J[2,2]]])
B_a = np.concatenate((zeros, B21_a), axis=0)

#Translational Dynamics

A12_t = np.eye(3)

A_up_t = np.concatenate((zeros, A12_t), axis=1)
A_down_t = np.concatenate((zeros, zeros), axis=1)
A_t = np.concatenate((A_up_t, A_down_t), axis=0)

B21_t = np.array([[G, 0, 0],
                    [0, -G, 0],
                    [0, 0, 1/M]])

B_t = np.concatenate((zeros, B21_t), axis=0)

#Weight Matrices

#Attitude

Q_a = np.eye(6)*10

R_a = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])*0.1

#Translational

Q_t = np.eye(6)*10

R_t = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])*2


#Optimal Gains

#Translational
P_t = solve_lqr(A_t, B_t, Q_t, R_t)
K_t = -np.linalg.inv(R_t)@B_t.T@P_t

#Rotational
P_a = solve_lqr(A_a, B_a, Q_a, R_a)
K_a = -np.linalg.inv(R_a)@B_a.T@P_a

#Get initial states
x_atual, _ = quad_model.reset(np.array([0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0]))

print(x_atual)

pos_atual = np.array([[x_atual[0,0], x_atual[0,2], x_atual[0,4], x_atual[0,1], x_atual[0,3], x_atual[0,5]]]).T
q_atual = np.array([[x_atual[0,6], x_atual[0,7], x_atual[0,8], x_atual[0,9]]]).T
eta_atual = np.array([[x_atual[0,10], x_atual[0,11], x_atual[0,12]]]).T

# q_atual = np.array([[1, 0, 0, 0]]).T
# eta_atual = np.array([[0, 0, 0]]).T

pos_list = []
quat_list = []
ang_vel_list = []

for i in range(outer_length):

    #Position Control
    pos_ref = np.array([[x_ref[i], y_ref[i], z_ref[i], dotx_ref[i], doty_ref[i], dotz_ref[i]]]).T
    # print(pos_ref, pos_atual)
    ang = np.array([[0], [0], [psiInt[i]]])
    quat_z = euler_quat(ang).reshape(4,1)
    # print(quat_z)
    T, q_ref = controller.pos_control(pos_atual, pos_ref, quat_z, K_t)

    # print('-------------------------------------------------------------------------------')
    # print(T)
    # print('-------------------------------------------------------------------------')

    for j in range(inner_length):

        taux, tauy, tauz = controller.att_control(q_atual, eta_atual, q_ref, K_a)
        # print(q_ref.T)
        # print(q_atual.T)
        # print('*********************************')
        action = np.array([T + M*G, taux, tauy, tauz])
        x, _, _ = quad_model.step(action)
        
        pos_atual = np.array([[x[0], x[2], x[4], x[1], x[3], x[5]]]).T
        q_atual = np.array([[x[6], x[7], x[8], x[9]]]).T
        eta_atual = np.array([[x[10], x[11], x[12]]]).T

        print("Posição:", pos_atual.T)
        print("Atitude:", q_atual.T)
     

        # print(q_atual.T)

    pos_list.append(pos_atual)
    quat_list.append(q_atual)
    ang_vel_list.append(eta_atual)        


t = np.arange(0, 201, 1)

pos_states = np.asarray(pos_list).reshape(201, 6)
quat_states = np.asarray(quat_list).reshape(201,4)
ang_vel = np.asarray(ang_vel_list).reshape(201,3)


#Attitude Controller Test


# qd = np.array([[0.996, 0.087, 0, 0]]).T

# for i in range(0, 50):

#     taux, tauy, tauz = controller.att_control(q_atual, eta_atual, qd, K_a)

#     action = np.array([M*G, taux, tauy, tauz])
    
#     x, _, _ = quad_model.step(action)
    
#     q_atual = np.array([[x[6], x[7], x[8], x[9]]]).T
#     print('q atual:', q_atual.T)
#     eta_atual = np.array([[x[10], x[11], x[12]]]).T
#     print('Velocidade Angular: ', eta_atual.T)
#     quat_list.append(q_atual)
#     ang_vel_list.append(eta_atual)   
#     # print(q_atual.T)

# t = np.arange(0, 50, 1)
# quat_states = np.asarray(quat_list).reshape(50,4)
# ang_vel = np.asarray(ang_vel_list).reshape(50,3)




# #Plot states
fig1, (x, y, z) = plt.subplots(3, 1, figsize=(15,15))
x.plot(t, pos_states[:,0], 'r', label=r'$x(t)$')
y.plot(t, pos_states[:,1], 'b', label=r'$y(t)$')
z.plot(t, pos_states[:,2], 'g', label=r'$z(t)$')
x.plot(t, x_ref, 'r--', label=r'$x_{ref}(t)$')
y.plot(t, y_ref, 'b--', label=r'$y_{ref}(t)$')
z.plot(t, z_ref, 'g--', label=r'$z_{ref}(t)$')

x.grid()
y.grid()
z.grid()

x.legend()
y.legend()
z.legend()


fig2, (dx, dy, dz) = plt.subplots(3, 1, figsize=(15,15))
dx.plot(t, pos_states[:,3], 'r', label=r'$\dot{x}(t)$')
dy.plot(t, pos_states[:,4], 'b', label=r'$\dot{y}(t)$')
dz.plot(t, pos_states[:,5], 'g', label=r'$\dot{z}(t)$')
dx.plot(t, dotx_ref, 'r--', label=r'$\dot{x}_{ref}(t)$')
dy.plot(t, doty_ref, 'b--', label=r'$\dot{y}_{ref}(t)$')
dz.plot(t, dotz_ref, 'g--', label=r'$\dot{z}_{ref}(t)$')

dx.grid()
dy.grid()
dz.grid()

dx.legend()
dy.legend()
dz.legend()

fig3, (q0, q1, q2, q3) = plt.subplots(4, 1, figsize=(15,15))
q0.plot(t, quat_states[:,0], 'r', label=r'$q_{0}(t)$')
q1.plot(t, quat_states[:,1], 'b', label=r'$q_{1}(t)$')
q2.plot(t, quat_states[:,2], 'g', label=r'$q_{2}(t)$')
q3.plot(t, quat_states[:,3], 'y', label=r'$q_{3}(t)$')
# q0.plot(t, quat_states[:,0], 'r', label=r'$q_{0}(t)$')
# q1.plot(t, quat_states[:,1], 'b', label=r'$q_{1}(t)$')
# q2.plot(t, quat_states[:,2], 'g', label=r'$q_{2}(t)$')
# q3.plot(t, quat_states[:,3], 'y', label=r'$q_{3}(t)$')

q0.grid()
q1.grid()
q2.grid()
q3.grid()

q0.legend()
q1.legend()
q2.legend()
q3.legend()

fig4, (nx, ny, nz) = plt.subplots(3, 1, figsize=(15,15))
nx.plot(t, ang_vel[:,0], 'b', label=r'$p(t)$')
ny.plot(t, ang_vel[:,1], 'g', label=r'$q(t)$')
nz.plot(t, ang_vel[:,2], 'y', label=r'$r(t)$')
# nx.plot(t, ang_vel[:,0], 'b', label=r'$p(t)$')
# ny.plot(t, ang_vel[:,1], 'g', label=r'$q(t)$')
# nz.plot(t, ang_vel[:,2], 'y', label=r'$r(t)$')

nx.grid()
ny.grid()
nz.grid()


nx.legend()
ny.legend()
nz.legend()

# #Plot states
# fig2, (x, y, z, vx, vy, vz) = plt.subplots(6, 1, figsize=(15,15))
# x.plot(t, x_ref, 'r', label=r'$x(t)$')
# y.plot(t, y_ref, 'b', label=r'$y(t)$')
# z.plot(t, z_ref, 'g', label=r'$z(t)$')
# vx.plot(t, dotx_ref, 'r', label=r'$\dot{x}(t)$')
# vy.plot(t, doty_ref, 'b', label=r'$\dot{y}(t)$')
# vz.plot(t, dotz_ref, 'g', label=r'$\dot{z}(t)$')
# quat.plot(t, quat_states[:,0], 'r', label=r'$q_{0}(t)$')
# quat.plot(t, quat_states[:,1], 'b', label=r'$q_{1}(t)$')
# quat.plot(t, quat_states[:,2], 'g', label=r'$q_{2}(t)$')
# quat.plot(t, quat_states[:,3], 'y', label=r'$q_{3}(t)$')
# devquat.plot(t, ang_vel[:,0], 'b', label=r'$p(t)$')
# devquat.plot(t, ang_vel[:,1], 'g', label=r'$q(t)$')
# devquat.plot(t, ang_vel[:,2], 'y', label=r'$r(t)$')

# pos.grid()
# vel.grid()
# quat.grid()
# devquat.grid()

# pos.legend()
# vel.legend()
# quat.legend()
# devquat.legend()

plt.show()