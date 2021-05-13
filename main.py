from quadrotor_env import quad
from quadrotor_control import Controller
from scipy.linalg import solve_continuous_are as solve_lqr
import numpy as np
import matplotlib.pyplot as plt


#Quadrotor Model Object
quad_model = quad(t_step=0.01, n=1, training=False, euler=0, direct_control=0, T=1, clipped=False)
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
controller = Controller(total_time = 10, sample_time = 0.01, inner_length = inner_length)
x_ref, dotx_ref, ddotx_ref, y_ref, doty_ref, ddoty_ref, z_ref, dotz_ref, ddotz_ref = controller.trajectory_generator(radius=2, frequency=np.pi/4, max_h=5, min_h=0)
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
                [0, 0, 1]])*10

#Translational

Q_t = np.eye(6)*100

R_t = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])


#Optimal Gains

#Translational
P_t = solve_lqr(A_t, B_t, Q_t, R_t)
K_t = -np.linalg.inv(R_t)@B_t.T@P_t

#Rotational
P_a = solve_lqr(A_a, B_a, Q_a, R_a)
K_a = -np.linalg.inv(R_a)@B_a.T@P_a

#Get initial states
x_atual, _ = quad_model.reset()

# pos_atual = np.array([[x_atual[0,0], x_atual[0,2], x_atual[0,4], x_atual[0,1], x_atual[0,3], x_atual[0,5]]]).T
q_atual = np.array([[x_atual[0,6], x_atual[0,7], x_atual[0,8], x_atual[0,9]]]).T
eta_atual = np.array([[x_atual[0,10], x_atual[0,11], x_atual[0,12]]]).T

# q_atual = np.array([[1, 0, 0, 0]]).T
# eta_atual = np.array([[0, 0, 0]]).T

pos_list = []
quat_list = []
ang_vel_list = []

# for i in range(outer_length):

#     #Position Control
#     pos_ref = np.array([[x_ref[i], y_ref[i], z_ref[i], dotx_ref[i], doty_ref[i], dotz_ref[i]]]).T
#     T, q_ref = controller.pos_control(pos_atual, pos_ref, Q_t, R_t)

#     for j in range(inner_length):

#         action = controller.att_control(T, q_atual, q_ref, eta_atual, Q_a, R_a)
        
#         x, _, _ = quad_model.step(action)
        
#         pos_atual = np.array([[x[0,0], x[0,2], x[0,4], x[0,1], x[0,3], x[0,5]]]).T
#         q_atual = np.array([[x[0,6], x[0,7], x[0,8], x[0,9]]]).T
#         eta_atual = np.array([[x[0,10], x[0,11], x[0,12]]]).T

#         pos_list.append(pos_atual)
#         quat_list.append(q_atual)
#         ang_vel_list.append(eta_atual)        


# t = np.arange(0, 1020, 1)

# pos_states = np.asarray(pos_list).reshape(1020, 6)
# quat_states = np.asarray(quat_list).reshape(1020,4)
# ang_vel = np.asarray(ang_vel_list).reshape(1020,3)


#Attitude Controller Test

qd = np.array([[0.707, 0, 0, 0.707]]).T

for i in range(0, 1000):

    taux, tauy, tauz = controller.att_control(q_atual, eta_atual, qd, K_a)

    action = np.array([M*G, taux, tauy, tauz])
    
    x, _, _ = quad_model.step(action)
    
    q_atual = np.array([[x[6], x[7], x[8], x[9]]]).T
    print('q atual:', q_atual.T)
    eta_atual = np.array([[x[10], x[11], x[12]]]).T
    print('Velocidade Angular: ', eta_atual.T)
    quat_list.append(q_atual)
    ang_vel_list.append(eta_atual)   
    # print(q_atual.T)

t = np.arange(0, 1000, 1)
quat_states = np.asarray(quat_list).reshape(1000,4)
ang_vel = np.asarray(ang_vel_list).reshape(1000,3)

# #Plot states
fig, (quat, devquat) = plt.subplots(2, 1, figsize=(15,15))
# pos.plot(t, pos_states[:,0], 'r', label=r'$x(t)$')
# pos.plot(t, pos_states[:,1], 'b', label=r'$y(t)$')
# pos.plot(t, pos_states[:,2], 'g', label=r'$z(t)$')
# vel.plot(t, pos_states[:,3], 'r', label=r'$\dot{x}(t)$')
# vel.plot(t, pos_states[:,4], 'b', label=r'$\dot{y}(t)$')
# vel.plot(t, pos_states[:,5], 'g', label=r'$\dot{z}(t)$')
quat.plot(t, quat_states[:,0], 'r', label=r'$q_{0}(t)$')
quat.plot(t, quat_states[:,1], 'b', label=r'$q_{1}(t)$')
quat.plot(t, quat_states[:,2], 'g', label=r'$q_{2}(t)$')
quat.plot(t, quat_states[:,3], 'y', label=r'$q_{3}(t)$')
devquat.plot(t, ang_vel[:,0], 'b', label=r'$p(t)$')
devquat.plot(t, ang_vel[:,1], 'g', label=r'$q(t)$')
devquat.plot(t, ang_vel[:,2], 'y', label=r'$r(t)$')

# pos.grid()
# vel.grid()
quat.grid()
devquat.grid()

# pos.legend()
# vel.legend()
quat.legend()
devquat.legend()

plt.show()