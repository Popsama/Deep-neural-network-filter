# author: SaKuRa Pop
# data: 2021/4/9 9:37
import numpy as np
import matplotlib.pyplot as plt

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


def predict(X_pre, P_pre, U):
    X_priori = A @ X_pre.reshape(2, 1) + B@U.reshape(2, 1)
    P_priori = A @ P_pre @ A.transpose() + process_noise_variance_matrix
    return X_priori, P_priori


def modify(X_priori, P_priori, Z_next):
    K_gain = P_priori @ H.transpose() @ np.linalg.pinv(H@P_priori+measure_noise_variance_matrix)
    X_predict = X_priori + K_gain @ (Z_next.reshape(2, 1) - H@X_priori)
    P_pre = (np.identity(2) - K_gain@H)@P_priori
    return X_predict, P_pre


def Kalman_filter(X_pre, P_pre, U, Z_next):
    X_prior, P_prior = predict(X_pre, P_pre, U)
    X_predict, P_pre = modify(X_prior, P_prior, Z_next)
    return X_predict, P_pre


X_prediction = np.zeros_like(Z)

X_prev = np.array([[1],
                   [0]])

P_prev = np.array([[1, 0],
                   [0, 1]])

for i in range(1111):
    X_prev, P_prev = Kalman_filter(X_prev, P_prev, U[:, i], Z[:, i])
    X_prediction[:, i] = X_prev.reshape(2,)

print(X_prediction.shape)

plt.figure()
plt.plot(Z[0], label="measure")
plt.plot(X_prediction[0], label="kalman filter")
plt.plot(CH4_no_noise[500], label="actual")
plt.legend()
plt.show()