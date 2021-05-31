from operator import pos
from numpy.core.defchararray import array
from quadrotor_env import quad
from quadrotor_control import Controller
from quaternion_euler_utility import euler_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Quadrotor Model Object
quad_model = quad(t_step=0.01, n=1, training=False, euler=0, direct_control=0, T=1, clipped=True)
# seed = quad_model.seed(2)



#Trajectory Generator ---> Obtain positions, velocities and accelerations references vectors
inner_length = 10
controller = Controller(total_time = 10, sample_time = 0.01, inner_length = inner_length)

x_wp = np.array([[0, 0, 0]]).T
y_wp = np.array([[0, 0, 0]]).T
z_wp = np.array([[0, 0.5, 1]]).T
psi_wp = np.array([[0, np.pi, 2*np.pi]]).T

t = [0, 1, 2]
step = 0.01

_, _, x_matrix = controller.getCoeff_snap(x_wp, t)
_, _, y_matrix = controller.getCoeff_snap(y_wp, t)
_, _, z_matrix = controller.getCoeff_snap(z_wp, t)

x_ref, dotx_ref, ddotx_ref, _, _ = controller.evaluate_equations_snap(t, step, x_matrix)
y_ref, doty_ref, ddoty_ref, _, _ = controller.evaluate_equations_snap(t, step, y_matrix)
z_ref, dotz_ref, ddotz_ref, _, _ = controller.evaluate_equations_snap(t, step, z_matrix)


# x_ref, dotx_ref, ddotx_ref, y_ref, doty_ref, ddoty_ref, z_ref, dotz_ref, ddotz_ref, psiInt = controller.trajectory_generator(radius=1, frequency=np.pi/15, max_h=7, min_h=4)
outer_length = len(x_ref)


#Get initial states
# x_atual, _ = quad_model.reset(np.array([1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0]))

# ang = quad_model.ang
# ang_vel = quad_model.ang_vel

# pos_atual = np.array([[x_atual[0,0], x_atual[0,2], x_atual[0,4]]]).T
# vel_atual = np.array([[x_atual[0,1], x_atual[0,3], x_atual[0,5]]]).T
# ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
# ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T


# pos_list = []
# quat_list = []
# ang_vel_list = []


# Qa = np.array([[15, 0, 0, 0, 0, 0],
#                 [0, 15, 0, 0, 0, 0],
#                 [0, 0, 5, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 5]])*150
    
# Ra = np.diag(np.ones(3))*30

# At, Bt, Aa, Ba = controller.linearized_matrices(False)
# Ka = controller.LQR_gain(Aa, Ba, Qa, Ra)


# Qt = np.array([[15e-7, 0, 0, 0, 0, 0],
#                 [0, 15e-7, 0, 0, 0, 0],
#                 [0, 0, 15e-7, 0, 0, 0],
#                 [0, 0, 0, 2.15, 0, 0],
#                 [0, 0, 0, 0, 2.15, 0],
#                 [0, 0, 0, 0, 0, 2.15]])*300
    
# Rt = np.diag(np.ones(3))*10

# Kt = controller.LQR_gain(At, Bt, Qt, Rt)

# for i in range(outer_length):

#     #Position Control
#     pos_ref = np.array([[x_ref[i], y_ref[i], z_ref[i]]]).T
#     vel_ref = np.array([[dotx_ref[i], doty_ref[i], dotz_ref[i]]]).T
#     accel_ref = np.array([[ddotx_ref[i], ddoty_ref[i], ddotz_ref[i]]]).T
#     # print(pos_ref, pos_atual)
#     psi = 0
#     # print(quat_z)
#     # T, phi_des, theta_des = controller.pos_control(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi, Kt)

#     T, phi_des, theta_des = controller.pos_control_PD(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi)
    
#     ang_des = np.array([[float(phi_des), float(theta_des), float(psi)]]).T

#     for j in range(inner_length):

#         taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, Ka)

#         action = np.array([float(T), taux, tauy, tauz])
#         x, _, _ = quad_model.step(action)
        
#         ang = quad_model.ang
#         ang_vel = quad_model.ang_vel

#         pos_atual = np.array([[x[0], x[2], x[4]]]).T
#         vel_atual = np.array([[x[1], x[3], x[5]]]).T
#         ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
#         ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T

#     # print("Posição:", pos_atual.T)
#     # print("Atitude Atual:", ang_atual.T)
     

#         # print(q_atual.T)

#     pos_list.append(pos_atual)
#     quat_list.append(ang_atual)
#     ang_vel_list.append(ang_vel_atual)        


# t = np.arange(0, 101, 1)

# pos_states = np.asarray(pos_list).reshape(101, 3)
# quat_states = np.asarray(quat_list).reshape(101,3)
# ang_vel = np.asarray(ang_vel_list).reshape(101,3)


#Attitude Controller Test


# qd = np.array([[0.996, 0.087, 0, 0]]).T
# ang_des = np.array([[0.174533, 0.174533, 0]]).T

# Qa = np.array([[15, 0, 0, 0, 0, 0],
#                 [0, 15, 0, 0, 0, 0],
#                 [0, 0, 5, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 5]])*150
    
# Ra = np.diag(np.ones(3))*40

# _, _, Aa, Ba = controller.linearized_matrices(False)
# K = controller.LQR_gain(Aa, Ba, Qa, Ra)

# for i in range(0, 100):

#     taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, K)

#     _, _, _ = controller.att_control_PD(ang_atual, ang_vel_atual, ang_des)

#     action = np.array([9.81*1.03, taux, tauy, tauz])
    
#     x, _, _ = quad_model.step(action)
    
#     ang = quad_model.ang
#     ang_vel = quad_model.ang_vel
#     # q_atual = np.array([[x[6], x[7], x[8], x[9]]]).T
#     # print('q atual:', q_atual.T)
#     ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
#     ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T
#     print('Atitude:', ang_atual.T)
#     print('Velocidade Angular: ', ang_vel_atual.T)
#     quat_list.append(ang_atual)
#     ang_vel_list.append(ang_vel_atual)   
#     # print(q_atual.T)

# t = np.arange(0, 100, 1)
# quat_states = np.asarray(quat_list).reshape(100,3)
# ang_vel = np.asarray(ang_vel_list).reshape(100,3)

# print(quat_states[:,0])

t1 = np.arange(0, t[-1], 0.01)


fig, (p, v, a) = plt.subplots(3,1, figsize=(9,9))
p.plot(t1, x_ref)
v.plot(t1, dotx_ref)
a.plot(t1, ddotx_ref)
p.grid()
v.grid()
a.grid()


fig2, (p2, v2, a2) = plt.subplots(3,1, figsize=(9,9))
p2.plot(t1, y_ref)
v2.plot(t1, doty_ref)
a2.plot(t1, ddoty_ref)
p2.grid()
v2.grid()
a2.grid()


fig3, (p3, v3, a3) = plt.subplots(3,1, figsize=(9,9))
p3.plot(t1, z_ref)
v3.plot(t1, dotz_ref)
a3.plot(t1, ddotz_ref)

p3.grid()
v3.grid()
a3.grid()


plt.show()