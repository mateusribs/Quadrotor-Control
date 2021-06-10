import numpy as np
from quat_utils import Quat2Rot, QuatProd, computeAngles, SkewMat, DerivQuat
from numpy.linalg import inv, det
import scipy as sci
from scipy.spatial.transform import Rotation as R



class MEKF():

    def __init__(self, quad_position, sensor, cv_flag):
        
        self.quad_position = quad_position
        self.camera_meas_flag = cv_flag
        self.roll_est_list, self.pitch_est_list, self.yaw_est_list = [], [], []
        self.roll_real_list, self.pitch_real_list, self.yaw_real_list = [], [], []
        self.sensor = sensor

        #MEKF Constants
        q_k0, R0 = self.sensor.triad()
        self.q_k = np.array([[q_k0[1], q_k0[2], q_k0[3], q_k0[0]]], dtype='float32').T

        self.var_a = 0.0035**2
        self.var_c = 0.0023**2
        self.var_v = 0.035**2
        self.var_u = 0.00015**2
        self.b_k = np.array([[0, 0, 0]], dtype='float32').T
        # self.P_k = np.array([[0.1, 0, 0, 0, 0, 0],
        #                      [0, .1, 0, 0, 0, 0],
        #                      [0, 0, .1, 0, 0, 0],
        #                      [0, 0, 0, .1, 0, 0],
        #                      [0, 0, 0, 0, .01, 0],
        #                      [0, 0, 0, 0, 0, .01]])*1000
        self.P_k = np.eye(6)*1000
        self.dx_k = np.array([[0, 0, 0, 0, 0, 0]], dtype='float32').T
        self.dt = self.quad_position.t_step
    
        
    
    #Multiplicative Extended Kalman Filter
    def MEKF(self):
        
        
        #Measurement Update, Quaternion Update and Bias Update
        q_K, b_K, P_K = self.Update_MEKF(self.dx_k, self.q_k, self.b_k, self.P_k)
        #Predict Quaternion using Gyro Measuments
        q_k, P_k = self.Prediction_MEKF(q_K, P_K, b_K)

        #Reset and propagation
        self.q_k = q_k
        self.b_k = b_K
        self.P_k = P_k
        self.dx_k = np.array([[0, 0, 0, 0, 0, 0]], dtype='float32').T


        #Get Real Attitude from Quadrotor
        #Get Real Attitude from Quadrotor
        quat = self.quad_position.state[6:10]
        quat_real = np.array([quat[1], quat[2], quat[3], quat[0]])
        r_real = sci.spatial.transform.Rotation.from_quat(quat_real).as_euler('XYZ')

        roll_real = r_real[0]
        pitch_real = r_real[1]
        yaw_real = r_real[2]
        

        #Get Euler Angles from Estimated MEKF Quaternion

        r_est = sci.spatial.transform.Rotation.from_quat(self.q_k.flatten()).as_euler('XYZ')
        self.roll_est = r_est[0]
        self.pitch_est = r_est[1]
        self.yaw_est = r_est[2]

        self.roll_est_list.append(self.roll_est)
        self.pitch_est_list.append(self.pitch_est)
        self.yaw_est_list.append(self.yaw_est)

        self.roll_real_list.append(roll_real)
        self.pitch_real_list.append(pitch_real)
        self.yaw_real_list.append(yaw_real)

    def Prediction_MEKF(self, q_K, P_K, b_K):

        gyro = self.sensor.gyro()

        # Discrete Quaternion Propagation
        omega_hat_k = gyro - b_K
        omega_hat_k_norm = np.linalg.norm(omega_hat_k)

        # dot_q = DerivQuat(omega_hat_k, q_K)
        # q_kp1 = q_K + dot_q*self.dt
        # recipNorm = 1/(np.linalg.norm(q_kp1))
        # q_kp1 *= recipNorm
        # print('Euler:', q_kp1)

        Psi_K = (omega_hat_k/omega_hat_k_norm)*np.sin(0.5*omega_hat_k_norm*self.dt)  
          
        Omega22 = np.array([[np.cos(0.5*omega_hat_k_norm*self.dt)]], dtype='float32')
        Omega21 = -Psi_K.T
        Omega12 = Psi_K
        Omega11 = np.cos(0.5*omega_hat_k_norm*self.dt)*np.eye(3) - SkewMat(Psi_K)
        Omega_up = np.concatenate((Omega11, Omega12), axis=1)
        Omega_down = np.concatenate((Omega21, Omega22), axis=1)
        Omega = np.concatenate((Omega_up, Omega_down), axis=0)
        q_kp1 =  Omega@q_K
        q_kp1 *= 1/(np.linalg.norm(q_kp1))

        #Discrete error-state transition matrix

        Phi11 = np.eye(3) - (np.sin(omega_hat_k_norm*self.dt)/omega_hat_k_norm)*SkewMat(omega_hat_k) + SkewMat(omega_hat_k)@SkewMat(omega_hat_k)*((1 - np.cos(omega_hat_k_norm*self.dt)/(omega_hat_k_norm**2)))

        Phi12 = ((1 - np.cos(omega_hat_k_norm*self.dt)/(omega_hat_k_norm**2)))*SkewMat(omega_hat_k) - np.eye(3)*self.dt - ((omega_hat_k_norm*self.dt - np.sin(omega_hat_k_norm*self.dt))/(omega_hat_k_norm**3))*SkewMat(omega_hat_k)@SkewMat(omega_hat_k)

        Phi21 = np.zeros((3,3))
        Phi22 = np.eye(3)

        Phi_up = np.concatenate((Phi11, Phi12), axis=1) 
        Phi_down = np.concatenate((Phi21, Phi22), axis=1)
        Phi = np.concatenate((Phi_up, Phi_down), axis=0)
        
        #Discrete input matrix

        Gamma = np.array([[-1, 0, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        #Discrete noise process covariance matrix

        Q11 = (self.var_v*self.dt + (1/3)*self.var_u*self.dt**3)*np.eye(3)
        Q12 = 0.5*self.var_u*self.dt**2*np.eye(3)
        Q21 = Q12
        Q22 = self.var_u*self.dt*np.eye(3)
        Q_up = np.concatenate((Q11, Q12), axis=1)
        Q_down = np.concatenate((Q21, Q22), axis=1)
        Q = np.concatenate((Q_up, Q_down), axis=0)

        #Discrete Covariance Propagation
        P_kp1 = Phi@P_K@Phi.T + Gamma@Q@Gamma.T

        return q_kp1, P_kp1

    def Update_MEKF(self, dx_k, q_k, b_k, P_k):
        
        #Estimated Rotation Matrix - Global to Local - Inertial to Body
        A_q = Quat2Rot(q_k)


        if self.camera_meas_flag:
            #Current Sensor Measurement

            #Accelerometer
            self.induced_acceleration = self.quad_position.f_in.flatten()/1.03 - (A_q @ np.array([[0, 0, -9.82]]).T).flatten()
            accel_grav = self.sensor.accel() - self.induced_acceleration
            b_a = np.array([accel_grav], dtype='float32').T
            b_a *= 1/(np.linalg.norm(b_a))

            #Camera
            b_c = cam_vec
            #Measurement Sensor Vector
            b = np.concatenate((b_a, b_c), axis=0)

            #Reference Accelerometer Direction (Gravitational Acceleration)
            r_a = np.array([[0, 0, -1]]).T
            #Reference Camera Direction
            r_c = np.array([[1, 0, 0]]).T
            #Camera Estimated Output
            b_hat_c = A_q @ r_c
            #Accelerometer Estimated Output
            b_hat_a = A_q @ r_a
            #Estimated Measurement Vector
            b_hat = np.concatenate((b_hat_a, b_hat_c), axis=0)

            #Sensitivy Matrix Accelerometer
            H_ka_left = SkewMat(b_hat_a)
            H_ka_right = np.zeros((3,3))
            H_ka = np.concatenate((H_ka_left, H_ka_right), axis=1)
            
            #Sensitivy Matrix Camera   
            H_kc_left = SkewMat(b_hat_c)
            H_kc_right = np.zeros((3,3))
            H_kc = np.concatenate((H_kc_left, H_kc_right), axis=1)

            #General Sensitivy Matrix
            H_k = np.concatenate((H_ka, H_kc), axis=0)

            #Accelerometer Measurement Covariance Matrix
            Ra = np.array([[self.var_a, 0, 0],
                            [0, self.var_a, 0],
                            [0, 0, self.var_a]], dtype='float32')
            
            
            #Camera Measurement Covariance Matrix
            Rc = np.array([[self.var_c, 0, 0],
                            [0, self.var_c, 0],
                            [0, 0, self.var_c]], dtype='float32')
            
            #General Measurement Covariance Matrix
            R_top = np.concatenate((Ra, np.zeros((3,3))), axis=1)
            R_down = np.concatenate((np.zeros((3,3)), Rc), axis=1)
            R = np.concatenate((R_top, R_down), axis=0) 

            #Kalman Gain
            K_k = P_k@H_k.T @ inv(H_k@P_k@H_k.T + R)

            #Update Covariance
            P_K = (np.eye(6) - K_k@H_k)@P_k

            #Inovation (Residual)
            e_k = b - b_hat

            #Update State
            dx_K = K_k@e_k
        
        else:
            #Current Sensor Measurement

            #Accelerometer 
            induced_acceleration = self.quad_position.f_in.flatten()/1.03 - (A_q @ np.array([[0, 0, -9.82]]).T).flatten()
            accel_grav = self.sensor.accel() - induced_acceleration
            b_a = np.array([accel_grav], dtype='float32').T
            b_a *= 1/(np.linalg.norm(b_a))

            #Reference Accelerometer Direction (Gravitational Acceleration)
            r_a = np.array([[0, 0, -1]]).T
            #Accelerometer Estimated Output
            b_hat_a = A_q @ r_a

            #Sensitivy Matrix Accelerometer
            H_ka_left = SkewMat(b_hat_a)
            H_ka_right = np.zeros((3,3))
            H_ka = np.concatenate((H_ka_left, H_ka_right), axis=1)

            #Accelerometer Measurement Covariance Matrix
            Ra = np.array([[self.var_a, 0, 0],
                           [0, self.var_a, 0],
                           [0, 0, self.var_a]], dtype='float32')

            #Kalman Gain
            K_k = P_k@H_ka.T @ inv(H_ka@P_k@H_ka.T + Ra)

            #Update Covariance
            P_K = (np.eye(6) - K_k@H_ka)@P_k

            #Inovation (Residual)
            e_k = b_a - b_hat_a
  
            #Update State
            dx_K = K_k@e_k
        
        #Update Quaternion
        dq_k = dx_K[0:3,:]
        q_K = q_k + DerivQuat(dq_k, q_k)
        q_K *= 1/(np.linalg.norm(q_K))    

        #Update Biases
        db = dx_K[3:6,:]
        b_K = b_k + db

        # print(q_K)

        return q_K, b_K, P_K

    