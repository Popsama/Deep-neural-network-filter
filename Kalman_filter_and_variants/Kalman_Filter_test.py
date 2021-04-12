# author: SaKuRa Pop
# data: 2021/4/6 9:49
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(2)
process_noise_mean = 0  # process noise mean
process_noise_variance_matrix = np.array([[0.1, 0],  # process noise variance matrix
                                          [0, 0.1]])

w_1_std = np.sqrt(process_noise_variance_matrix[0][0])
w_2_std = np.sqrt(process_noise_variance_matrix[1][1])
print(np.sqrt(process_noise_variance_matrix[0][0]))
print(np.sqrt(process_noise_variance_matrix[1][1]))

white_noise_1 = np.random.normal(process_noise_mean, w_1_std, size=30)
white_noise_2 = np.random.normal(process_noise_mean, w_2_std, size=30)

print(white_noise_1.mean())
print(white_noise_1.std())
print(white_noise_2.mean())
print(white_noise_2.std())

W = np.hstack((white_noise_1.reshape(30, 1), white_noise_2.reshape((30, 1))))
print("W shape ", W.shape)
print("W matrix \n", W)
print("=======================================")
# x_1 = 0  # initial location
# x_2 = 1  # initial velocity
#
# X_pre = np.array([0, 1])
# X_pre = X_pre.reshape(2, 1)
# # state transition matrix
A = np.array([[1, 1],
              [0, 1]])
#
# X = np.zeros_like(W)
# X[0] = np.array(([0, 1]))
# # state space equation
# for i in range(29):
#     cache = A @ X_pre + W[i].reshape(2, 1)
#     X_pre = cache
#     X[i+1] = cache.reshape(1, 2)
X = np.array([[0.000, 1.000],
              [0.919, 0.715],
              [1.968, 0.676],
              [2.843, 1.105],
              [3.666, 1.197],
              [5.415, 0.878],
              [6.317, 0.740],
              [7.012, 1.032],
              [8.302, 1.647],
              [9.973, 1.651],
              [11.849,1.641],
              [13.468,2.120],
              [15.826,2.076],
              [17.762,1.962],
              [19.939,1.712],
              [21.914,1.715],
              [23.186,1.734],
              [24.769,1.756],
              [26.619,2.143],
              [29.297,2.265],
              [32.245,2.437],
              [34.603,2.189],
              [36.591,2.413],
              [38.303,1.931],
              [39.930,2.063],
              [41.955,1.779],
              [43.787,1.801],
              [45.469,1.762],
              [47.371,2.539],
              [50.265,2.786],
              [53.014,2.640]])
print("X matrix \n", X)
print("X shape", X.shape)
print("=======================================")

H = np.array([[1, 0],
              [0, 1]])
measure_noise_mean = 0  # process noise mean
measure_noise_variance_matrix = np.array([[1, 0],  # process noise variance matrix
                                          [0, 1]])
Z = np.array([[0, 0],
              [1.986, 2.535],
              [4.320, 1.185],
              [3.397, 2.001],
              [4.627, 2.130],
              [3.725, 0.736],
              [7.059, -1.203],
              [6.820, 0.929],
              [8.448, 0.822],
              [10.014,3.799],
              [12.154,1.258],
              [12.093,3.294],
              [15.562,2.343],
              [17.607,1.393],
              [20.424,2.604],
              [21.988,0.936],
              [22.856,1.348],
              [24.913,2.553],
              [25.566,2.351],
              [29.118,2.667],
              [33.036,2.913],
              [35.831,1.818],
              [35.654,1.142],
              [38.414,3.262],
              [38.903,0.946],
              [41.185,2.477],
              [44.511,2.363],
              [45.007,2.386],
              [47.676,3.102],
              [50.266,4.478],
              [53.994,1.968]])
# v_1_std = np.sqrt(measure_noise_variance_matrix[0][0])
# v_2_std = np.sqrt(measure_noise_variance_matrix[1][1])
# print(np.sqrt(measure_noise_variance_matrix[0][0]))
# print(np.sqrt(measure_noise_variance_matrix[1][1]))
#
# v_noise_1 = np.random.normal(measure_noise_mean, w_1_std, size=29)
# v_noise_2 = np.random.normal(measure_noise_mean, w_2_std, size=29)
# V = np.hstack((v_noise_1.reshape(29, 1), v_noise_2.reshape((29, 1))))
# print("V shape", V.shape)
# Z = np.zeros_like(X)
#
# for i in range(1, 30):
#     cache = H @ X[i].reshape(2, 1) + V[i-1].reshape(2, 1)
#     Z[i] = cache.reshape(1, 2)
# print("Z \n", Z)
# print("z shape", Z.shape)
# print("=======================================")

# initial error covariance matrix P
P = np.array([[2, 0],
              [0, 2]])


def predict(X_pre, P_pre):
    X_priori = A @ X_pre.reshape(2, 1)
    P_priori = A @ P_pre @ A.transpose() + process_noise_variance_matrix
    return X_priori, P_priori


def modify(X_priori, P_priori, Z_next):
    K_gain = P_priori @ H.transpose() @ np.linalg.inv(H@P_priori+measure_noise_variance_matrix)
    X_predict = X_priori + K_gain @ (Z_next.reshape(2, 1) - H@X_priori)
    P_pre = (np.identity(2) - K_gain@H)@P_priori
    return X_predict, P_pre


def Kalman_filter(X_pre, P_pre, Z_next):
    X_prior, P_prior = predict(X_pre, P_pre)
    X_predict, P_pre = modify(X_prior, P_prior, Z_next)
    return X_predict, P_pre


X_prediction = np.zeros_like(X)
X_prediction[0] = X[0]
X_prev = X[0]
P_prev = P
for i in range(30):
    X_prev, P_prev = Kalman_filter(X_prev, P_prev, Z[i+1])
    X_prediction[i+1] = X_prev.reshape(1, 2)


plt.figure()
plt.title("location")
plt.plot(X[1:, 0], c="red", label="actual location")
plt.plot(Z[1:, 0], c="blue", label="measured location")
plt.plot(X_prediction[1:, 0], c="fuchsia", label="optimal predicted location")
plt.legend()

plt.figure()
plt.title("velocity")
plt.plot(X[1:, 1], c="red", label="actual velocity")
plt.plot(Z[1:, 1], c="blue", label="measured velocity")
plt.plot(X_prediction[1:, 1], c="fuchsia", label="optimal predicted velocity")
plt.legend()

plt.figure()
plt.title("velocity")
plt.plot(Z[1:, 1], c="blue", label="measured velocity")
plt.plot(X_prediction[1:, 1], c="fuchsia", label="optimal predicted velocity")
plt.legend()

plt.show()



