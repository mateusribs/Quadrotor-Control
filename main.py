from quadrotor_model import QuadrotorModel
import numpy as np
import matplotlib.pyplot as plt


t0 = 0
tf = 20
N = 100
x0 = np.array([0, 0, 100, 0, 0, 0, 0.707, 0, 0, 0.707, 0, 0, 0])
u0 = np.array([100*np.pi/2, 100*np.pi/2, 100*np.pi/2, 100*np.pi/2], dtype='float32')

quadrotor = QuadrotorModel(t0, tf, N)
x, t = quadrotor.runge_kutta(quadrotor.quat_dynamics, x0, u0)




#Plot states
fig, (pos, vel, quat, devquat) = plt.subplots(4, 1, figsize=(9,7))
pos.plot(t, x[:,0], 'r', label=r'$x(t)$')
pos.plot(t, x[:,1], 'b', label=r'$y(t)$')
pos.plot(t, x[:,2], 'g', label=r'$z(t)$')
vel.plot(t, x[:,3], 'r', label=r'$\dot{x}(t)$')
vel.plot(t, x[:,4], 'b', label=r'$\dot{y}(t)$')
vel.plot(t, x[:,5], 'g', label=r'$\dot{z}(t)$')
quat.plot(t, x[:,6], 'r', label=r'$q_{0}(t)$')
quat.plot(t, x[:,7], 'b', label=r'$q_{1}(t)$')
quat.plot(t, x[:,8], 'g', label=r'$q_{2}(t)$')
quat.plot(t, x[:,9], 'y', label=r'$q_{3}(t)$')
devquat.plot(t, x[:,10], 'b', label=r'$p(t)$')
devquat.plot(t, x[:,11], 'g', label=r'$q(t)$')
devquat.plot(t, x[:,12], 'y', label=r'$r(t)$')

pos.grid()
vel.grid()
quat.grid()
devquat.grid()

pos.legend()
vel.legend()
quat.legend()
devquat.legend()

plt.show()