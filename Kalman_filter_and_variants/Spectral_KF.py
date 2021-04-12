# author: SaKuRa Pop
# data: 2021/4/12 13:35
import numpy as np
import matplotlib.pyplot as plt


def baseline_regularization(original_signal, baseline_signal):
    regularization = original_signal / baseline_signal
    return regularization


CH4_simulated_no_noise_npy = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_sim_no_noise.npy"
CH4_no_noise = np.load(CH4_simulated_no_noise_npy)  # (1000, 1111) row for data column for number of data
baseline = CH4_no_noise[0]
CH4_no_noise = baseline_regularization(CH4_no_noise, baseline)

CH4_simulated_noisy_npy = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_sim_withnoise.npy"
CH4_noisy = np.load(CH4_simulated_noisy_npy)  # (1000, 1111) row for data column for number of data, with noise
CH4_noisy = baseline_regularization(CH4_noisy, baseline)

"""我们假定使用第500个数据，也即500ppm的CH4的吸收谱作为sample"""
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(CH4_no_noise[500])
plt.ylim(0.95, 1.05)

plt.subplot(1, 2, 2)
plt.plot(CH4_noisy[500])
plt.ylim(0.95, 1.05)

process_var = 0.009858427**2  # process noise variance
sensor_var = 0.009223176**2  # measurement noise variance

Z = CH4_noisy[500]
nk = np.arange(0, 1111)
Z = np.vstack((Z, nk))  # (2, 1111) measurement matrix

U = CH4_no_noise[500]
ones = np.ones((1111,))
U = np.vstack((U, ones))  # (2, 1111) control matrix


A = np.array([[0, 0],
              [0, 1]])

B = np.array([[1, 0],
              [0, 1]])

H = np.array([[1, 0],
              [0, 1]])

process_noise_variance_matrix = np.array([[process_var, 0],
                                          [0, 0]])

measure_noise_variance_matrix = np.array([[sensor_var],
                                          [0]])


class SpectralKalmanFilter:
    """
    plain Kalman Filter类，初始化方法中包含了气体吸收光谱系统空间状态方程所需要的
    A：状态转移矩阵
    B：控制矩阵
    H：测量系统参数矩阵
    U：系统控制矩阵
    X_initial:初始X值 （2，1）维度
    P_initial:初始P值 （2，1）维度
    Z：系统测量值（实际值）
    Q&R:系统过程、测量噪声协方差矩阵（通过实验计算得到的）

    predict方法计算 X_hat_priori_k = A·X_hat_(k-1) + B·U_k;
                   P_priori_k = A·P_(k-1)·AT + Q
    modify方法计算 Kg_k; X_hat_k; P_k
    filtering方法计算从0到1111index的全部信号的的kalman filtering之后的optimal estimated values
    """
    def __init__(self, u, z, w, v):
        self.A = np.array([[0, 0],
                           [0, 1]])
        self.B = np.array([[1, 0],
                           [0, 1]])
        self.H = np.array([[1, 0],
                           [0, 1]])
        self.U = u
        self.Z = z
        self.Q = np.array([[w, 0],
                           [0, 0]])
        self.R = np.array(([[v],
                            [0]]))
        self.X_prediction = np.zeros_like(self.Z)
        self.X_prev = np.array([[1],
                                [0]])
        self.P_prev = np.array([[1, 0],
                                [0, 1]])

    def predict(self, X_pre, P_pre, U):
        X_priori = self.A @ X_pre.reshape(2, 1) + self.B @ U.reshape(2, 1)
        P_priori = self.A @ P_pre @ A.transpose() + self.Q
        return X_priori, P_priori

    def modify(self, X_priori, P_priori, Z_next):
        K_gain = P_priori @ self.H.transpose() @ np.linalg.pinv(self.H @ P_priori + self.R)
        X_predict = X_priori + K_gain @ (Z_next.reshape(2, 1) - self.H @ X_priori)
        P_pre = (np.identity(2) - K_gain @ self.H) @ P_priori
        return X_predict, P_pre

    def Kalman_filter(self, X_pre, P_pre, U, Z_next):
        X_prior, P_prior = self.predict(X_pre, P_pre, U)
        X_predict, P_pre = self.modify(X_prior, P_prior, Z_next)
        return X_predict, P_pre

    def filtering(self):
        for i in range(1111):
            self.X_prev, self.P_prev = self.Kalman_filter(self.X_prev, self.P_prev, self.U[:, i], self.Z[:, i])
            self.X_prediction[:, i] = self.X_prev.reshape(2,)
        return self.X_prediction


Filter = SpectralKalmanFilter(U, Z, process_var, sensor_var)
x_prediction = Filter.filtering()
print(x_prediction.shape)
plt.figure()
plt.plot(Z[0], label="measure")
plt.plot(x_prediction[0], label="kalman filter")
plt.plot(CH4_no_noise[500], label="actual")
plt.legend()
plt.show()
