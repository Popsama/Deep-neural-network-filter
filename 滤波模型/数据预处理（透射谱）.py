# author: SaKuRa Pop
# data: 2021/1/8 11:11
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from FilterNet_utils import array_to_tensor, Filter, read_pickle_to_array, Data_set, training_1, training_2

"""数据文件路径"""
file_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\剪裁后PKL数据"
file_name_1 = r"\CH4_0dB.pkl"
file_name_2 = r"\CH4_10dB.pkl"
file_name_3 = r"\CH4_20dB.pkl"
file_name_4 = r"\C2H2_0dB.pkl"
file_name_5 = r"\C2H2_10dB.pkl"
file_name_6 = r"\C2H2_20dB.pkl"

""""加载数据"""
CH4_label = read_pickle_to_array(file_path, file_name_1)

print("CH4_label.shape :", CH4_label.shape)  # (10000, 2000, 1)
print("============================================================================")

CH4_10dB = read_pickle_to_array(file_path, file_name_2)
print("CH4_10dB.shape", CH4_10dB.shape)  # (10000, 1, 2000)
print("============================================================================")

CH4_20dB = read_pickle_to_array(file_path, file_name_3)
print("CH4_20dB.shape", CH4_20dB.shape)   # (10000, 1, 2000)
print("============================================================================")

print(CH4_label.shape[0])

linear_fitor = linear_model.LinearRegression()
X = np.arange(0, 2000).reshape(-1, 1)
Y = CH4_10dB[9999]
linear_fitor.fit(X, Y)

Y_predict = linear_fitor.predict(X)

plt.figure()
plt.subplot(2, 2, 1)
plt.scatter(X, Y)
plt.plot(X, Y_predict, c="orange")
plt.plot(X, CH4_label[9999], c="lightcoral")

plt.subplot(2, 2, 2)
correction = CH4_10dB[9999] / CH4_label[0]
plt.plot(correction)

correction2 = CH4_label[9999] / CH4_label[0]

plt.subplot(2, 2, 3)
plt.plot(Y_predict)
plt.plot(CH4_label[0])

plt.subplot(2, 2, 4)
plt.plot(correction)
plt.plot(correction2)


plt.show()


def baseline_regularization(original_signal, baseline_signal):
    regularization = original_signal / baseline_signal
    return regularization


correction_CH4_10dB = np.zeros_like(CH4_10dB)
correction_CH4_label = np.zeros_like(CH4_label)
correction_CH4_20dB = np.zeros_like(CH4_20dB)
for i in range(CH4_label.shape[0]):
    correction_CH4_label[i, :] = baseline_regularization(CH4_label[i], CH4_label[0])
    correction_CH4_10dB[i, :] = baseline_regularization(CH4_10dB[i], CH4_label[0])
    correction_CH4_20dB[i, :] = baseline_regularization(CH4_20dB[i], CH4_label[0])

np.save(r"./correction_CH4_label.npy", correction_CH4_label)
np.save(r"./correction_CH4_10dB.npy", correction_CH4_10dB)
np.save(r"./correction_CH4_20dB.npy", correction_CH4_20dB)
print("saved finished")