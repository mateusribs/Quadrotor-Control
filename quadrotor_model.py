import numpy as np
from quaternion_euler_utility import deriv_quat

class QuadrotorModel:

    #Mass
    m = 0.2
    #Gravity
    g = 9.81
    #Inertia Tensor
    I_xx = 16.83*10**-3
    I_yy = 16.83*10**-3
    I_zz = 28.34*10**-3
    #Arm Length
    L = 0.2
    #
    kT = 1.435*10**-5
    kD = 2.4086*10**-7
    J = 5*10**-5

    def __init__(self, initial_time, final_time, max_iterations):
        
        self.t0 = initial_time
        self.tf = final_time
        self.max_i = max_iterations

    def runge_kutta(self, f, x0, u0):
    #4th order Runge Kutta Numerical Integrator
        t = np.linspace(self.t0, self.tf, self.max_i+1)
        x = np.array((self.max_i+1)*[x0])
        h = t[1]-t[0]
        for i in range(self.max_i):
            k1 = h * f(x[i], u0)
            k2 = h * f(x[i] + 0.5 * k1, u0)
            k3 = h * f(x[i] + 0.5 * k2, u0)
            k4 = h * f(x[i] + k3, u0)
            x[i+1] = x[i] + (k1 + 2*(k2 + k3) + k4) / 6
        return x, t
    
    def quat_dynamics(self, x0, omega):
        
        omega_1, omega_2, omega_3, omega_4 = omega

        #Quaternions
        q0 = float(x0[6])
        q1 = float(x0[7])
        q2 = float(x0[8])
        q3 = float(x0[9])

        #Total Trust
        T = self.kT*(omega_1**2 + omega_2**2 + omega_3**2 + omega_4**2)

        #Translational Dynamics
        dot_x = float(x0[3])
        ddot_x = 2*T*(q0*q2 + q1*q3)/self.m
        dot_y = float(x0[4])
        ddot_y = -2*T*(q0*q1 - q2*q3)/self.m
        dot_z = float(x0[5])
        ddot_z = T*(q0**2 - q1**2 - q2**2 + q3**2)/self.m - self.g
        #Rotational Dynamics
        dot_quat = deriv_quat(np.array([x0[10], x0[11], x0[12]]), np.array([q0, q1, q2, q3]))
        dot_p = self.L*self.kT*(omega_2**2 - omega_4**2) - self.J*x0[10]*(-omega_1 + omega_2 - omega_3 + omega_4)
        dot_q = self.L*self.kT*(omega_1**1 - omega_3**2) - self.J*x0[11]*(omega_1 - omega_2 + omega_3 - omega_4)
        dot_r = self.kD*(-omega_1**2 + omega_2**2 - omega_3**2 + omega_4**2)

        return np.array([dot_x, dot_y, dot_z, ddot_x, ddot_y, ddot_z, dot_quat[0],
                            dot_quat[1], dot_quat[2], dot_quat[3], dot_p, dot_q, dot_r])
        

    def quat_dynamics_linearized(self):
        pass

    def euler_dynamics(self):
        pass
    
    def euler_dynamics_linearized(self):
        pass
        
