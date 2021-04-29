import numpy as np
from scipy.linalg import solve_continuous_are as solve_lqr
from quadrotor_env import quad
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
plt.rcParams.update({'font.size': 15})



#Integrador Numérico de Runge-Kutta 4ª Ordem
def rk4(f, u0, t0, tf , n):
    t = np.linspace(t0, tf, n+1)
    u = np.array((n+1)*[u0])
    h = t[1]-t[0]
    for i in range(n):
        k1 = h * f(u[i], t[i], i)
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h, i)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h, i)
        k4 = h * f(u[i] + k3, t[i] + h, i)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6
    return u, t

def trajectory_generator(t):
    '''This method creates the trajectory for a drone to follow'''

    r = 2
    f = 0.025
    height_i = 0
    height_f = 5
    d_height=height_f-height_i

    # Define the x, y, z dimensions for the drone trajectories
    alpha=2*np.pi*f*t

    # Trajectory 1
    x=r*np.cos(alpha)
    y=r*np.sin(alpha)
    z=height_i+d_height/(t[-1])*t

    x_dot=-r*np.sin(alpha)*2*np.pi*f
    y_dot=r*np.cos(alpha)*2*np.pi*f
    z_dot=d_height/(t[-1])*np.ones(len(t))

    x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
    y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
    z_dot_dot=0*np.ones(len(t))

    # Vector of x and y changes per sample time
    dx=x[1:len(x)]-x[0:len(x)-1]
    dy=y[1:len(y)]-y[0:len(y)-1]
    dz=z[1:len(z)]-z[0:len(z)-1]

    dx=np.append(np.array(dx[0]),dx)
    dy=np.append(np.array(dy[0]),dy)
    dz=np.append(np.array(dz[0]),dz)


    return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot

def quat_prod(p, q):

    ps = p[0]
    pv = np.array([p[1], p[2], p[3]])
    qs = q[0]
    qv = np.array([q[1], q[2], q[3]])

    quat0 = ps*qs - pv.T@qv
    quatV = ps*qv + qs*pv + np.cross(pv, qv, axis=0)
    quatRes = np.concatenate((quat0, quatV), axis=0)

    return quatRes

def LQR(x_atual, x_desired, A, B, Q, R):

    #State Error
    x_error = x_atual - x_desired
    print('erro:', x_error)
    #Optimal Gain
    P = solve_lqr(A, B, Q, R)
    K = np.linalg.inv(R)@B.T@P

    #Optimal Input
    u = -K@x_error

    return u[0], u[1], u[2]



#Linearized Matrices

zeros = np.zeros((3,3))


#Constants
J = np.array([[16.83*10**-3, 0, 0],
              [0, 16.83*10**-3, 0],
              [0, 0, 28.34*10**-3]])
#Mass and Gravity
M, G = 1.03, 9.82

KT = 1.435*10**-5
KD = 2.4086*10**-7
L = 0.26
#Attitude Dynamics

A12_a = np.eye(3)*0.5

A_up_a = np.concatenate((zeros, A12_a), axis=1)
A_down_a = np.concatenate((zeros, zeros), axis=1)
A_a = np.concatenate((A_up_a, A_down_a), axis=0)

B21_a = np.array([[1/J[0,0], 0 , 0],
                  [0, 1/J[1,1], 0],
                  [0, 0, 1/J[2,2]]])
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

Q_a = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])*10000

R_a = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

#Translational

Q_t = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])


R_t = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])*10

#---------------------------------------------------------


#Quadrotor Model

innerDyn_length = 4
Ts = 0.1
t = np.arange(0,100+Ts*innerDyn_length,Ts*innerDyn_length) # time from 0 to 100 seconds, sample time (Ts=0.4 second)
x_ref, x_dot_ref, x_dot_dot_ref, y_ref, y_dot_ref, y_dot_dot_ref, z_ref, z_dot_ref, z_dot_dot_ref = trajectory_generator(t)
plotl=len(t) # Number of outer control loop iterations


t_step = 0.01
time = 250
quad = quad(t_step, time, training = False, direct_control=0, T=1)
quad.seed(1)
# env_plot = plotter(quad, True, False)

#Initial States - X, Y, Z, Xdot, Ydot, Zdot, qx, qy, qz, omegax, omegay, omegaz
# states = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# statesTotal = [states]
# att_states = []
trans_states = np.array([x_ref, y_ref, z_ref, x_dot_ref, y_dot_ref, z_dot_ref],dtype='float32')
# print(trans_states[:,0])

state, action = quad.reset()

ref_states = np.array([[0, 0, 0, 0, 0, 0]],dtype='float32').T



for i in range(0, innerDyn_length):
    att_states = np.array([[quad.state[7], quad.state[8], quad.state[9], quad.state[10], quad.state[11], quad.state[12]]],dtype='float32').T
    #Attitude Control
    taux, tauy, tauz = LQR(att_states, ref_states, A_a, B_a, Q_a, R_a)

    action = np.array([dT, float(taux), float(tauy), float(tauz)])

    _, _,  done = quad.step(action)

    print(att_states)
        

#Plot
# fig, (pos, vel) = plt.subplots(2, 1, figsize=(9,7), sharex=True)
# pos.plot(t, x_ref, 'b', alpha=0.6, lw=2, label=r'$x(t)$')
# pos.plot(t, x[:,1], 'r', alpha=0.6, lw=2, label=r'$y(t)$')
# pos.plot(t, x[:,2], 'g',alpha=0.6, lw=2, label=r'$z(t)$')
# vel.plot(t, x[:,3], 'b--', alpha=0.6, lw=2, label=r'$\dot{x}(t)$')
# vel.plot(t, x[:,4], 'r--', alpha=0.6, lw=2, label=r'$\dot{y}(t)$')
# vel.plot(t, x[:,5], 'g--',alpha=0.6, lw=2, label=r'$\dot{z}(t)$')
# pos.set_title('Posição Quadrirrotor')
# vel.set_xlabel('Tempo(s)')
# pos.set_ylabel('Posição (m)')
# vel.set_ylabel('Velocidade (m/s)')
# pos.grid()
# pos.legend()
# vel.grid()
# vel.legend()

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.scatter3D(x_ref, y_ref, z_ref)



# plt.show()


# for i in range(0, time):

#     # #Position Control
#     pos_states = np.array([quad.state[0], quad.state[2], quad.state[4], 0, 0, 0],dtype='float32').T
#     uxd, uyd, dT = LQR(pos_states, trans_states[:,i], A_t, B_t, Q_t, R_t)

#     uxyd_mod = np.sqrt(uxd**2 + uyd**2)
#     alpha = np.arcsin(uxyd_mod)

#     q0d = np.cos(alpha/2)
#     q1d = uyd/(2*q0d)
#     q2d = uxd/(2*q0d)

#     qpd = np.array([[q0d, q1d, q2d, 0]],dtype='float32').T
#     qzd = np.array([[1, 0, 0, 0]],dtype='float32').T
#     qd = quat_prod(qpd, qzd)
#     qdv = qd[1:4]

#     ref_states = np.concatenate((qdv, np.zeros((3,1))), axis=0)