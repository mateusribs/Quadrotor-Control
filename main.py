from operator import pos
from numpy.core.defchararray import array
from quadrotor_env import quad, sensor
from quadrotor_control import Controller
from quadrotor_estimator import MEKF
from quaternion_euler_utility import euler_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Quadrotor Model Object
quad_model = quad(t_step=0.01, n=1, training=False, euler=0, direct_control=0, T=1, clipped=True)
sens = sensor(quad_model)

# seed = quad_model.seed(2)



#Trajectory Generator ---> Obtain positions, velocities and accelerations references vectors
inner_length = 1
controller = Controller(total_time = 10, sample_time = 0.01, inner_length = inner_length)

x_wp = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T
y_wp = np.array([[0, 0, 0.2, 0.3, 0.3]]).T
z_wp = np.array([[2, 2.5, 3, 3, 3]]).T
psi_wp = np.array([[0, 0, 0, np.pi/4, np.pi/2]]).T

t = [0, 5, 10, 15, 20]
step = 0.01

_, _, x_matrix = controller.getCoeff_snap(x_wp, t)
_, _, y_matrix = controller.getCoeff_snap(y_wp, t)
_, _, z_matrix = controller.getCoeff_snap(z_wp, t)
_, _, psi_matrix = controller.getCoeff_accel(psi_wp, t)

x_ref, dotx_ref, ddotx_ref, _, _ = controller.evaluate_equations_snap(t, step, x_matrix)
y_ref, doty_ref, ddoty_ref, _, _ = controller.evaluate_equations_snap(t, step, y_matrix)
z_ref, dotz_ref, ddotz_ref, _, _ = controller.evaluate_equations_snap(t, step, z_matrix)
psi_ref, _, _ = controller.evaluate_equations_accel(t, step, psi_matrix)

# x_ref, dotx_ref, ddotx_ref, y_ref, doty_ref, ddoty_ref, z_ref, dotz_ref, ddotz_ref, psiInt = controller.trajectory_generator(radius=1, frequency=np.pi/15, max_h=7, min_h=4)
outer_length = len(x_ref)

#Get initial states
x_atual, _ = quad_model.reset(np.array([0.2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0]))

sens.reset()

ang = quad_model.ang
ang_vel = quad_model.ang_vel

pos_atual = np.array([[x_atual[0,0], x_atual[0,2], x_atual[0,4]]]).T
vel_atual = np.array([[x_atual[0,1], x_atual[0,3], x_atual[0,5]]]).T

ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T


pos_list = []
vel_list = []
ang_list = []
ang_vel_list = []
ang_ref_list = []
T_list = []


# Qa = np.array([[1, 0, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0, 0],
#                 [0, 0, 0, 10**-3, 0, 0],
#                 [0, 0, 0, 0, 10**-3, 0],
#                 [0, 0, 0, 0, 0, 10**-4]])*1
    
# Ra = np.array([[1, 0, 0],
#                [0, 1, 0],
#                [0, 0, 1]])*1

# At, Bt, Aa, Ba = controller.linearized_matrices(False)
# Ka = controller.LQR_gain(Aa, Ba, Qa, Ra)



# Qt = np.array([[1, 0, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 1]])

# Rt = np.array([[1, 0, 0],
#                [0, 1, 0],
#                [0, 0, 1]])*1

# Kt = controller.LQR_gain(At, Bt, Qt, Rt)

for i in range(outer_length):

    #Position Control
    pos_ref = np.array([[x_ref[i], y_ref[i], z_ref[i]]]).T
    vel_ref = np.array([[dotx_ref[i], doty_ref[i], dotz_ref[i]]]).T
    accel_ref = np.array([[ddotx_ref[i], ddoty_ref[i], ddotz_ref[i]]]).T
    # print(pos_ref, pos_atual)
    psi = psi_ref[i]
    # print(quat_z)
    # T, phi_des, theta_des = controller.pos_control(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi, Kt)

    T, phi_des, theta_des = controller.pos_control_PD(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi)

    ang_des = np.array([[float(phi_des), float(theta_des), float(psi)]]).T

    for j in range(inner_length):
        
        # mekf = MEKF(quad_model, sens, False)
        ang = quad_model.ang
        ang_vel = quad_model.ang_vel
        # mekf.MEKF()
        # ang_atual = np.array([[mekf.roll_est, mekf.pitch_est, ang[2]]]).T
        # ang_vel_atual = sens.gyro()

        ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
        ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T

        # taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, Ka)
        taux, tauy, tauz = controller.att_control_PD(ang_atual, ang_vel_atual, ang_des)
        # taux, tauy, tauz = controller.att_control_quaternion_PD(quat_atual, quat_des, vel_ang_atual)

        action = np.array([float(T), taux, tauy, tauz])
        x, _, _ = quad_model.step(action)
        
        ang = quad_model.ang
        ang_vel = quad_model.ang_vel

        pos_atual = np.array([[x[0], x[2], x[4]]]).T
        vel_atual = np.array([[x[1], x[3], x[5]]]).T

        

        # ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
        # ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T

    print("Posição:", pos_atual.T)
    # print("Empuxo:", T)
    # print("Phi:", phi_des)
    # print("Theta:", theta_des)
    # print("Atitude Atual:", ang_atual.T)
     

        # print(q_atual.T)

    pos_list.append(pos_atual)
    vel_list.append(vel_atual)
    ang_list.append(ang_atual)
    ang_vel_list.append(ang_vel_atual)
    ang_ref_list.append(ang_des)
    T_list.append(T)        


t1 = np.arange(0, t[-1], step)

pos_states = np.asarray(pos_list).reshape(outer_length, 3)
vel_states = np.asarray(vel_list).reshape(outer_length, 3)
ang_states = np.asarray(ang_list).reshape(outer_length,3)
ang_vel = np.asarray(ang_vel_list).reshape(outer_length,3)
ang_refe = np.asarray(ang_ref_list).reshape(outer_length,3)


#Attitude Controller Test


# quat_des = np.array([[0.998, 0.044, -0.002, 0.044]]).T

############################## RADIANS ########################################
# ang_des = np.array([[0.1, 0.1, 0.01]]).T

# # Qa = np.array([[90, 0, 0, 0, 0, 0],
# #                 [0, 90, 0, 0, 0, 0],
# #                 [0, 0, 10, 0, 0, 0],
# #                 [0, 0, 0, 10**-3, 0, 0],
# #                 [0, 0, 0, 0, 10**-3, 0],
# #                 [0, 0, 0, 0, 0, 10**-4]])*0.2
    
# # Ra = np.array([[.085, 0, 0],
# #                [0, .085, 0],
# #                [0, 0, 1]])*0.1

# # _, _, Aa, Ba = controller.linearized_matrices(False)
# # K = controller.LQR_gain(Aa, Ba, Qa, Ra)

# ite = 20

# for i in range(0, ite):

#     mekf = MEKF(quad_model, sens, False)

#     # taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, K)
#     taux, tauy, tauz = controller.att_control_PD(ang_atual, ang_vel_atual, ang_des)

#     action = np.array([9.81*1.03, taux, tauy, tauz])
    
#     x, _, _ = quad_model.step(action)


#     ang = quad_model.ang
#     ang_vel = quad_model.ang_vel
#     # quat_atual = np.array([[x[6], x[7], x[8], x[9]]]).T
#     # vel_ang_atual = np.array([[x[10], x[11], x[12]]]).T
#     # print('q atual:', q_atual.T)

#     mekf.MEKF()
#     # ang_atual = np.array([[mekf.roll_est, mekf.pitch_est, ang[2]]]).T
#     quat_atual = np.array([mekf.q_k[3], mekf.q_k[0], mekf.q_k[1], [x[9]]])
#     # vel_ang_atual = sens.gyro()

#     ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
#     ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T
    

#     # print('Angulo Real:', ang_atual.T)
#     # print('Angulo Estimado:', ang_est)

#     ang_list.append(ang_atual)
#     ang_vel_list.append(ang_vel_atual)   
#     # print(q_atual.T)

# t1 = np.arange(0, ite, 1)
# ang_states = np.asarray(ang_list).reshape(ite,3)
# ang_vel = np.asarray(ang_vel_list).reshape(ite,3)
# ang_refe = ang_des.T*np.ones((ite, 3))

# print(quat_states[:,0])


# print(x_ref)

fig, (p, v) = plt.subplots(2,1, figsize=(9,9))
p.plot(t1, x_ref, 'y--')
p.plot(t1, pos_states[:,0], 'g')
v.plot(t1, dotx_ref, 'y--')
v.plot(t1, vel_states[:,0], 'b')
p.grid()
v.grid()


fig2, (p2, v2) = plt.subplots(2,1, figsize=(9,9))
p2.plot(t1, y_ref, 'y--')
p2.plot(t1, pos_states[:, 1], 'g')
v2.plot(t1, doty_ref, 'y--')
v2.plot(t1, vel_states[:,1], 'b')
p2.grid()
v2.grid()


fig3, (p3, v3) = plt.subplots(2,1, figsize=(9,9))
p3.plot(t1, z_ref, 'y--')
p3.plot(t1, pos_states[:,2], 'g')
v3.plot(t1, dotz_ref, 'y--')
v3.plot(t1, vel_states[:,2], 'b')

p3.grid()
v3.grid()


fig4, (ph, th, ps) = plt.subplots(3, 1, figsize=(15,15))
ph.plot(t1, ang_states[:,0], 'r')
th.plot(t1, ang_states[:,1], 'g')
ps.plot(t1, ang_states[:,2], 'b')

ph.plot(t1, ang_refe[:,0], 'r--')
th.plot(t1, ang_refe[:,1], 'g--')
ps.plot(t1, ang_refe[:,2], 'b--')
# q3.plot(t1, ang_refe[:,3], 'y--')

# vph.plot(t1, ang_vel[:,0], 'r')
# vth.plot(t1, ang_vel[:,1], 'g')
# vps.plot(t1, ang_vel[:,2], 'b')

# vph.grid()
# vth.grid()
# vps.grid()

ph.grid()
th.grid()
ps.grid()


# fig5, emp = plt.subplots(1,1,figsize=(10,10))
# emp.plot(t1, T_list)
# emp.grid()

# fig6 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x_ref, y_ref, z_ref, 'y')
# ax.plot3D(pos_states[:,0], pos_states[:,1], pos_states[:,2], 'g--')

plt.show()