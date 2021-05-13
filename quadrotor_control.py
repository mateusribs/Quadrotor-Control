import numpy as np
from quaternion_euler_utility import deriv_quat
from quaternion_euler_utility import quat_prod

class Controller:




    def __init__(self, total_time, sample_time, inner_length):
        
        self.total_t = total_time
        self.Ts = sample_time
        self.il = inner_length
        
    
    def trajectory_generator(self, radius, frequency, max_h, min_h):
        
        t = np.arange(0, self.total_t + self.Ts*self.il, self.Ts*self.il)

        d_height = max_h - min_h

        # Define the x, y, z dimensions for the drone trajectories
        alpha = 2*np.pi*frequency*t

        # Trajectory 1
        x = radius*np.cos(alpha)*0
        y = radius*np.sin(alpha)*0
        z = min_h + d_height/(t[-1])*t*0

        x_dot = -radius*np.sin(alpha)*2*np.pi*frequency*0
        y_dot = radius*np.cos(alpha)*2*np.pi*frequency*0
        z_dot = d_height/(t[-1])*np.ones(len(t))*0

        x_dot_dot = -radius*np.cos(alpha)*(2*np.pi*frequency)**2
        y_dot_dot = -radius*np.sin(alpha)*(2*np.pi*frequency)**2
        z_dot_dot = 0*np.ones(len(t))

        # Vector of x and y changes per sample time
        dx = x[1:len(x)]-x[0:len(x)-1]
        dy = y[1:len(y)]-y[0:len(y)-1]
        dz = z[1:len(z)]-z[0:len(z)-1]

        dx = np.append(np.array(dx[0]),dx)
        dy = np.append(np.array(dy[0]),dy)
        dz = np.append(np.array(dz[0]),dz)


        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot
    
    
    def pos_control(self, pos_atual, pos_desired, K):

        #Compute error
        pos_error = pos_atual - pos_desired

        #Compute Optimal Control Law
        u = -K@pos_error
        
        #Optimal Input
        uxd = float(u[0])
        uyd = float(u[1])
        dT = float(u[2])

        mod_uxyd = np.sqrt(uxd**2+uyd**2)
        alpha = np.arcsin(mod_uxyd)

        q0d = np.cos(alpha/2)
        q1d = uyd/(2*q0d)
        q2d = uxd/(2*q0d)
    

        #Desired quaternion
        qpd = np.array([[q0d, q1d, q2d, 0]],dtype='float32').T
        mod_qpd = np.linalg.norm(qpd)
        qzd = np.array([[1, 0, 0, 0]],dtype='float32').T
        qd = quat_prod(qpd, qzd)
        qdv = qd[1:4]

        return dT, qd
    
    def att_control(self, q_atual, eta_atual, q_des, K):
        

        #Compute error
        q_conj = np.concatenate((q_atual[0].reshape(1,1), -q_atual[1:4]), axis=0)
        q_error = quat_prod(q_conj, q_des)

        if q_error[0] < 0:
            q_error = -q_error

        eta_error = np.zeros((3,1)) - eta_atual

        att_error = np.concatenate((q_error[1:4], eta_error), axis=0)

        # print(att_error.T)
        #Compute Optimal Control Law
        u = -K@att_error

        #Optimal input
        tau_x = float(u[0])
        tau_y = float(u[1])
        tau_z = float(u[2])

        return tau_x, tau_y, tau_z
    
    def quat2axis(self, q):

        qdv_mod = np.linalg.norm(q[1:4])

        if qdv_mod != 0:
            theta = 2*(q[1:4]/qdv_mod)*np.arccos(q[0])
        elif qdv_mod == 0:
            theta = np.zeros((3,1))
        
        return theta

