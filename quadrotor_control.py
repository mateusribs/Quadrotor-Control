import numpy as np
from numpy.core.numeric import NaN
from quaternion_euler_utility import deriv_quat
from quaternion_euler_utility import quat_prod
from scipy.linalg import solve_continuous_are as solve_lqr


class Controller:

    #Constants
    J = np.array([[16.83*10**-3, 0, 0],
                  [0, 16.83*10**-3, 0],
                  [0, 0, 28.34*10**-3]])
    Ixx = 16.83*10**-3
    Iyy = 16.83*10**-3
    Izz = 28.34*10**-3
    #Mass and Gravity
    M, G = 1.03, 9.82

    KT = 1.435*10**-5
    KD = 2.4086*10**-7
    L = 0.26

    def __init__(self, total_time, sample_time, inner_length):
        
        self.total_t = total_time
        self.Ts = sample_time
        self.il = inner_length

        self.nd_ant = 1
        self.ed1_ant = 0
        self.ed2_ant = 0
        self.ed3_ant = 0
    
    def trajectory_generator(self, radius, frequency, max_h, min_h):
        
        t = np.arange(0, self.total_t + self.Ts*self.il, self.Ts*self.il)

        d_height = max_h - min_h

        # Define the x, y, z dimensions for the drone trajectories
        alpha = 2*np.pi*frequency*t

        # Trajectory 1
        x = radius*np.cos(alpha)
        y = radius*np.sin(alpha)*0
        z = min_h + d_height/(t[-1])*t

        x_dot = -radius*np.sin(alpha)*2*np.pi*frequency
        y_dot = radius*np.cos(alpha)*2*np.pi*frequency*0
        z_dot = d_height/(t[-1])*np.ones(len(t))

        x_dot_dot = -radius*np.cos(alpha)*(2*np.pi*frequency)**2
        y_dot_dot = -radius*np.sin(alpha)*(2*np.pi*frequency)**2*0
        z_dot_dot = 0*np.ones(len(t))

        # Vector of x and y changes per sample time
        dx = x[1:len(x)]-x[0:len(x)-1]
        dy = y[1:len(y)]-y[0:len(y)-1]
        dz = z[1:len(z)]-z[0:len(z)-1]

        dx = np.append(np.array(dx[0]),dx)
        dy = np.append(np.array(dy[0]),dy)
        dz = np.append(np.array(dz[0]),dz)

        # Define the reference yaw angles
        psi=np.zeros(len(x))
        psiInt=psi
        psi[0]=np.arctan2(y[0],x[0])+np.pi/2
        psi[1:len(psi)]=np.arctan2(dy[1:len(dy)],dx[1:len(dx)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]


        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psiInt

    ######################################### LQR CONTROL APPROACH ############################################################

    def linearized_matrices(self, quaternion = True):

        if quaternion:
            #Linearized System - Quaternion Approach

            zeros = np.zeros((3,3))

            #Attitude Dynamics

            A12_a = np.eye(3)*0.5

            A_up_a = np.concatenate((zeros, A12_a), axis=1)
            A_down_a = np.concatenate((zeros, zeros), axis=1)
            A_a = np.concatenate((A_up_a, A_down_a), axis=0)

            B21_a = np.array([[1/self.J[0,0], 0, 0], [0, 1/self.J[1,1], 0], [0, 0, 1/self.J[2,2]]])
            B_a = np.concatenate((zeros, B21_a), axis=0)

            #Translational Dynamics

            A12_t = np.eye(3)

            A_up_t = np.concatenate((zeros, A12_t), axis=1)
            A_down_t = np.concatenate((zeros, zeros), axis=1)
            A_t = np.concatenate((A_up_t, A_down_t), axis=0)

            B21_t = np.array([[self.G, 0, 0],
                                [0, -self.G, 0],
                                [0, 0, 1/self.M]])

            B_t = np.concatenate((zeros, B21_t), axis=0)
        
        else:
            #Linearized System - Euler Approach

            zeros = np.zeros((3,3))

            #Attitude Dynamics

            A12_a = np.eye(3)

            A_up_a = np.concatenate((zeros, A12_a), axis=1)
            A_down_a = np.concatenate((zeros, zeros), axis=1)
            A_a = np.concatenate((A_up_a, A_down_a), axis=0)

            B21_a = np.array([[1/self.Ixx, 0, 0], [0, 1/self.Iyy, 0], [0, 0, 1/self.Izz]])
            B_a = np.concatenate((zeros, B21_a), axis=0)

            #Translational Dynamics

            A12_t = np.eye(3)

            A_up_t = np.concatenate((zeros, A12_t), axis=1)
            A_down_t = np.concatenate((zeros, zeros), axis=1)
            A_t = np.concatenate((A_up_t, A_down_t), axis=0)

            B21_t = np.array([[self.G, 0, 0],
                                [0, -self.G, 0],
                                [0, 0, 1/self.M]])

            B_t = np.concatenate((zeros, B21_t), axis=0) 

        
        return A_t, B_t, A_a, B_a
    
    def LQR_gain(self, A, B, Q, R):

        P = solve_lqr(A, B, Q, R)
        K = -np.linalg.inv(R)@B.T@P
    
        return K

    def pos_control(self, pos_atual, pos_des, vel_atual, vel_des, accel_des, psi, K):

        #Compute error
        dpos_error = pos_des - pos_atual
        vel_error = vel_des - vel_atual

        n = vel_des/np.linalg.norm(vel_des)
        t = accel_des/np.linalg.norm(accel_des)
        b = np.cross(t, n, axis=0)

        if np.isnan(b).any:
            pos_error = dpos_error
        else:
            pos_error = (dpos_error.T@n)@n + (dpos_error.T@b)@b

        # print(pos_error.T)
        #Compute Optimal Control Law
        u = -K@np.concatenate((pos_error, vel_error), axis=0)
        
        T = self.M*(self.G + u[2])

        phi_des = (u[0]*np.sin(psi) - u[1]*np.cos(psi))/self.G
        theta_des = (u[0]*np.cos(psi) + u[1]*np.sin(psi))/self.G

        return T, phi_des, theta_des
    
    def att_control(self, ang_atual, ang_des, ang_vel_atual, K):

        phi = float(ang_atual[0])
        theta = float(ang_atual[1])
        psi = float(ang_atual[2])

        #Compute error
        
        ang_error = ang_des - ang_atual
        print("Erro angulo:", ang_error.T)
        ang_vel_error = np.zeros((3,1)) - ang_vel_atual

        #Compute Optimal Control Law
        T = np.array([[1/self.Ixx, np.sin(phi)*np.tan(theta)/self.Iyy, np.cos(phi)*np.tan(theta)/self.Izz],
                      [0, np.cos(phi)/self.Iyy, -np.sin(phi)/self.Izz],
                      [0, np.sin(phi)*np.cos(theta)/self.Iyy, np.cos(phi)*np.cos(theta)/self.Izz]])

        u = -K@np.concatenate((ang_error, ang_vel_error), axis=0)

        #Optimal input
        tau_x = float(u[0])
        tau_y = float(u[1])
        tau_z = float(u[2])

        return tau_x, tau_y, tau_z
        
    ######################################## PID CONTROL APPROACH #############################################################    

    def att_control_PD(self, ang_atual, ang_vel_atual, ang_des):
        
        phi = float(ang_atual[0])
        theta = float(ang_atual[1])
        psi = float(ang_atual[2])

        #PID gains unlimited
        Kp = np.array([[200, 0 ,0],
                       [0, 200, 0],
                       [0, 0, 100]])*8
        Kd = np.array([[50, 0, 0],
                       [0, 50, 0],
                       [0, 0, 35]])*1.1

        # Kp = np.array([[15, 0 ,0],
        #                [0, 15, 0],
        #                [0, 0, 1]])*500
        # Kd = np.array([[50, 0, 0],
        #                [0, 50, 0],
        #                [0, 0, 45]])*10
        
        
        angle_error = ang_des - ang_atual
        print('Erro angulo:', angle_error.T)
        ang_vel_error = np.zeros((3,1)) - ang_vel_atual
        #Compute Optimal Control Law

        T = np.array([[1/self.Ixx, np.sin(phi)*np.tan(theta)/self.Iyy, np.cos(phi)*np.tan(theta)/self.Izz],
                      [0, np.cos(phi)/self.Iyy, -np.sin(phi)/self.Izz],
                      [0, np.sin(phi)*np.cos(theta)/self.Iyy, np.cos(phi)*np.cos(theta)/self.Izz]])

        u = np.linalg.inv(T)@(Kp@angle_error + Kd@ang_vel_error)

        #Optimal input
        tau_x = float(u[0])
        tau_y = float(u[1])
        tau_z = float(u[2])

        return tau_x, tau_y, tau_z
    
    def pos_control_PD(self, pos_atual, pos_des, vel_atual, vel_des, accel_des, psi):

        #PD gains unlimited
        Kp = np.array([[4, 0 ,0],
                       [0, 2, 0],
                       [0, 0, 9.5]])*10.5
        Kd = np.array([[4.5, 0, 0],
                       [0, 4, 0],
                       [0, 0, 4]])*3.8

        # Kp = np.array([[200, 0 ,0],
        #                [0, 200, 0],
        #                [0, 0, 100]])*8
        # Kd = np.array([[50, 0, 0],
        #                [0, 50, 0],
        #                [0, 0, 35]])*1.1

        dpos_error = pos_des - pos_atual

        vel_error = vel_des - vel_atual

        n = vel_des/np.linalg.norm(vel_des)
        t = accel_des/np.linalg.norm(accel_des)
        b = np.cross(t, n, axis=0)

        if np.isnan(b).any:
            pos_error = dpos_error
        else:
            pos_error = (dpos_error.T@n)@n + (dpos_error.T@b)@b


        rddot_c = accel_des + Kd@vel_error + Kp@pos_error

        T = self.M*(self.G + rddot_c[2])

        phi_des = (rddot_c[0]*np.sin(psi) - rddot_c[1]*np.cos(psi))/self.G
        theta_des = (rddot_c[0]*np.cos(psi) + rddot_c[1]*np.sin(psi))/self.G

        return T, phi_des, theta_des



    #################################### TRAJECTORY PLANNER FUNCTIONS ######################################################
    
    #Returns the derivatives
    def polyT(self, n, k, t):

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

    #Get the optimal snap functions coefficients 
    def getCoeff_snap(self, waypoints, t):

        n = len(waypoints) - 1
        A = np.zeros((8*n, 8*n))
        b = np.zeros((8*n, 1))

        # print(b.T)

        row = 0
        #Initial constraints
        for i in range(0, 1):
            A[row, 8*(i):8*(i+1)] = self.polyT(8, 0, t[0])
            b[i, 0] = waypoints[0]
            row = row + 1
        
        for k in range(1, 4):
            A[row, 0:8] = self.polyT(8, k, t[0])
            row = row + 1
        
        if n == 1:
            #Last P constraints
            for i in range(0, 1):
                A[row, 8*(i):8*(i+1)] = self.polyT(8, 0, t[-1])
                b[row, 0] = waypoints[1]
                row = row + 1  
            
            for k in range(1, 4):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, k, t[-1])
                row = row + 1



        elif n>1:


            #Pi constraints
            shift = 0
            for j in range(1, n):
                
                
                
                for i in range(0, 2):
                    A[row, 8*(i+shift):8*(i+1+shift)] = self.polyT(8, 0, t[j])
                    b[row, 0] = waypoints[j]

                    row = row + 1

                for k in range(1, 7):
                    A[row, 8*(j-1):8*(j)] = self.polyT(8, k, t[j])
                    A[row, 8*(j):8*(j+1)] = -self.polyT(8, k, t[j])
                    row = row + 1
                
                shift += 1
            
            
            #Last P constraints
            for i in range(0, 1):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, 0, t[-1])
                b[row, 0] = waypoints[n]
                row = row + 1
            
            for k in range(1, 4):
                A[row, 8*(n) - 8:8*(n)] = self.polyT(8, k, t[-1])
                row = row + 1


        coeff = np.linalg.inv(A)@b
        
        c_matrix = coeff.reshape(n, 8)

        return A, b, c_matrix

    #Compute the snap trajectory equations at time 't'
    def equation_snap(self, t, c_matrix, eq_n):
        x = self.polyT(8, 0, t)
        v = self.polyT(8, 1, t)
        a = self.polyT(8, 2, t)
        j = self.polyT(8, 3, t)
        s = self.polyT(8, 4, t)
        

        P = np.sum(x*c_matrix[eq_n,:])
        V = np.sum(v*c_matrix[eq_n,:])
        A = np.sum(a*c_matrix[eq_n,:])
        J = np.sum(j*c_matrix[eq_n,:])
        S = np.sum(s*c_matrix[eq_n,:])
        

        return P, V, A, J, S
    
    #Storage the values of any equations at time 't' in lists
    def evaluate_equations_snap(self, t, step, c_matrix):
            
        skip = 0

        x_list = []
        v_list = []
        a_list = []
        j_list = []
        s_list = []

        for i in np.arange(0, t[-1], step):

            if skip == 0:

                if i >= t[skip] and i<=t[skip+1]:
                
                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

                else:

                    skip += 1

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

            elif skip > 0 and skip < len(t):

                if i > t[skip] and i <= t[skip+1]:

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)

                else:

                    skip += 1

                    p, v, a, j, s = self.equation_snap(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
                    j_list.append(j)
                    s_list.append(s)
        
        return x_list, v_list, a_list, j_list, s_list


    #Get the optimal acceleration functions coefficients 
    def getCoeff_accel(self, waypoints, t):

        n = len(waypoints) - 1
        A = np.zeros((4*n, 4*n))
        b = np.zeros((4*n, 1))

        # print(b.T)

        row = 0
        #Initial constraints
        for i in range(0, 1):
            A[row, 4*(i):4*(i+1)] = self.polyT(4, 0, t[0])
            b[i, 0] = waypoints[0]
            row = row + 1
        
        for k in range(1, 2):
            A[row, 0:4] = self.polyT(4, k, t[0])
            row = row + 1
        
        if n == 1:
            #Last P constraints
            for i in range(0, 1):
                A[row, 4*(i):4*(i+1)] = self.polyT(4, 0, t[-1])
                b[row, 0] = waypoints[1]
                row = row + 1  
            
            for k in range(1, 2):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, k, t[-1])
                row = row + 1



        elif n>1:


            #Pi constraints
            shift = 0
            for j in range(1, n):
                
                
                
                for i in range(0, 2):
                    A[row, 4*(i+shift):4*(i+1+shift)] = self.polyT(4, 0, t[j])
                    b[row, 0] = waypoints[j]

                    row = row + 1

                for k in range(1, 3):
                    A[row, 4*(j-1):4*(j)] = self.polyT(4, k, t[j])
                    A[row, 4*(j):4*(j+1)] = -self.polyT(4, k, t[j])
                    row = row + 1
                
                shift += 1
            
            
            #Last P constraints
            for i in range(0, 1):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, 0, t[-1])
                b[row, 0] = waypoints[n]
                row = row + 1
            
            for k in range(1, 2):
                A[row, 4*(n) - 4:4*(n)] = self.polyT(4, k, t[-1])
                row = row + 1


        coeff = np.linalg.inv(A)@b
        
        c_matrix = coeff.reshape(n, 4)

        return A, b, c_matrix

    #Compute the acceleration trajectory equations at time 't'
    def equation_accel(self, t, c_matrix, eq_n):
        x = self.polyT(4, 0, t)
        v = self.polyT(4, 1, t)
        a = self.polyT(4, 2, t)

        P = np.sum(x*c_matrix[eq_n,:])
        V = np.sum(v*c_matrix[eq_n,:])
        A = np.sum(a*c_matrix[eq_n,:])
        

        return P, V, A

    #Storage the values of any equations at time 't' in lists
    def evaluate_equations_accel(self, t, step, c_matrix):
            
        skip = 0

        x_list = []
        v_list = []
        a_list = []

        for i in np.arange(0, t[-1], step):

            if skip == 0:

                if i >= t[skip] and i<=t[skip+1]:
                
                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

                else:

                    skip += 1

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

            elif skip > 0 and skip < len(t):

                if i > t[skip] and i <= t[skip+1]:

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)

                else:

                    skip += 1

                    p, v, a = self.equation_accel(i, c_matrix, skip)

                    x_list.append(p)
                    v_list.append(v)
                    a_list.append(a)
        
        return x_list, v_list, a_list


