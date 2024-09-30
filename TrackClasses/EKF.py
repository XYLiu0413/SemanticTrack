import numpy as np
import math

__all__=['model_cv','EKF']
def model_cv(dt=0.5):
    """
    generate a constant velocity model
    dt,A,Q,R is constant
    """
    model = dict()
    model["dt"] = dt
    #Process transition matrix
    model["A"] = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    # convert errors to the number of range or doppler cells
    # model["scale_bins"] = np.array([1.0 / delta_r, 1.0 / delta_v])

    # Obervation matrix
    # Any value can be set here, because there is no meaning. The H matrix is transformed according to the prior state s in the update step.
    # Set H here to only define the functions of the CV model for the specification
    model["H"] = np.eye(3,4)

    #Process noise covariance matrix
    q = 4*np.eye(4)  #q is spectral density of the process white noise
    Q=np.array([[dt**3/3, 0, dt**2/2, 0],
              [0,   dt**3/3, 0, dt**2/2],
              [dt**2/2, 0,      dt,   0],
              [0,   dt**2/2,     0,  dt]])
    model["Q"] =np.dot(q,Q)  

    # observation noise matrix
    sigema_x=0.06   #range resolution：0.06m
    sigema_y=0.06
    sigema_vr=0.15# veloc resolution
    model["R"] = np.array([[sigema_x**2, 0,0],
                           [0, sigema_y**2,0],
                           [0,0,sigema_vr**2]])
    return model

class EKF(object):
    """
    EKF class，包括 predict and update
    """
    def __init__(self, model, s0=np.array([0, 0, 1, 1]), P0=np.eye(4)):
        self.s = s0              =
        #self.u = np.zeros((3,1)) #observations vector
        self.P = P0
        self.P_history=[P0]
        self.A = model["A"]  # State transition matrix
        self.H = model["H"]  # Observation matrix
        self.Q = model["Q"]  # Process noise covariance matrix
        self.R = model["R"]  # Observation noise covariance matrix


    def ekf_predict(self):
        # Prior state estimate
        self.s_apr = np.dot(self.A, self.s) 
        # Prior error covariance estimate
        self.P_apr = np.dot(np.dot(self.A,self.P),self.A.T)  + self.Q

    def ekf_innovation(self):
        """
        Must be executed before the ekf_update
        """

        # Update J_H matrix(jacobian matrix) at first
        # it will be used in calculating Kalman Gain
        # It will be used in Gating Function(before Update step)
        if self.s_apr[0] == 0 and self.s_apr[1] == 0:
            print("error!")
        else:
            fenzi = self.s_apr[2] * self.s_apr[1] - self.s_apr[3] * self.s_apr[0]
            fenmu = math.sqrt(self.s_apr[0] ** 2 + self.s_apr[1] ** 2)
            self.H[2][0] = self.s_apr[1] * fenzi / (fenmu ** 3)
            self.H[2][1] = self.s_apr[0] * -fenzi / (fenmu ** 3)
            self.H[2][2] = self.s_apr[0] / fenmu
            self.H[2][3] = self.s_apr[1] / fenmu
        # Compute innovation covariance
        self.C = self.H @ self.P_apr @ self.H.T + self.R 
        # 𝑯(S𝑎𝑝𝑟(𝑛)):  converts the predicted a-priori states 𝑠𝑎𝑝𝑟(𝑛) from state to observation
        # It will be used in Gating Function(before Update step)
        # So it's placed here, not in the ekf_update
        self.u_apr = np.array([self.s_apr[0], self.s_apr[1],
                            (self.s_apr[0] * self.s_apr[2] + self.s_apr[1] * self.s_apr[3]) / fenmu])

    def ekf_update(self, u):
        """
        :param u:  observations vector
        """

        # Kalman Gain
        K = self.P_apr @ self.H.T @ np.linalg.inv(self.C)
        # Posterior state estimate
        self.y = u -self.u_apr # Compute innovation (or measurement residual)
        self.s = self.s_apr + K @ self.y
        # Updated error covariance matrix
        self.P = self.P_apr - K @ self.H @ self.P_apr

        self.P_history.append(self.P)

if __name__ == '__main__':
    obeject1 = EKF(model_cv())
    object2=EKF
    print(obeject1.s)
