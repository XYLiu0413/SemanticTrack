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
    # 这里设置任何值都可，因为没有意义。H矩阵在更新步骤会根据先验状态s变换。
    # 这里设置H仅为规范定义该CV模型的函数
    model["H"] = np.eye(3,4)

    #Process noise covariance matrix
    q = 4*np.eye(4)  #q is spectral density of the process white noise
    Q=np.array([[dt**3/3, 0, dt**2/2, 0],
              [0,   dt**3/3, 0, dt**2/2],
              [dt**2/2, 0,      dt,   0],
              [0,   dt**2/2,     0,  dt]])
    model["Q"] =np.dot(q,Q)  #矩阵相乘用dot，*代表每个数值对应相乘

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
        self.s = s0              #无需专门设至为列向量，矩阵乘法的条件由dot函数自动调整满足
        #self.u = np.zeros((3,1)) #observations vector
        self.P = P0
        self.P_history=[P0]
        self.A = model["A"]  # State transition matrix
        self.H = model["H"]  # Observation matrix
        self.Q = model["Q"]  # Process noise covariance matrix
        self.R = model["R"]  # Observation noise covariance matrix


    def ekf_predict(self):
        # Prior state estimate
        self.s_apr = np.dot(self.A, self.s) #dot(或@)会自动将s转换为列向量与A进行矩阵乘法，但得出的结果s_apr仍为array形式的横向量。
        # Prior error covariance estimate
        self.P_apr = np.dot(np.dot(self.A,self.P),self.A.T)  + self.Q

    def ekf_innovation(self):
        """
        Must be executed before the ekf_update
        """

        # Update J_H matrix(雅可比矩阵) at first
        # it will be used in calculating Kalman Gain
        # It will be used in Gating Function(在Update之前执行)
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
        self.C = self.H @ self.P_apr @ self.H.T + self.R  # @表示矩阵相乘
        # 𝑯(S𝑎𝑝𝑟(𝑛)):  converts the predicted a-priori states 𝑠𝑎𝑝𝑟(𝑛) from state to observation
        # It will be used in Gating Function(在Update之前执行)
        # So it's placed here, not in the ekf_update
        self.u_apr = np.array([self.s_apr[0], self.s_apr[1],
                            (self.s_apr[0] * self.s_apr[2] + self.s_apr[1] * self.s_apr[3]) / fenmu])

    def ekf_update(self, u):
        """
        :param u:  observations vector，每帧的传感器测量量
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
