# author: SaKuRa Pop
# data: 2020/11/16 15:13
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
from FilterNet_utils import ConvBNReLU, pooling1d, global_average_pooling1d, Filter, read_pickle_to_array, Data_set, training_1, \
    training_2, Filter2, Filter3, Filter4, Filter5, array_to_tensor


def smoothing(array, threshold=0.5):
    anomaly_indices = np.where(array < threshold)
    array = np.delete(array, anomaly_indices)
    return array


def smoothing_n_comparision(array, label, threshold=0.5):
    anomaly_indices = np.where(array < threshold)
    array = np.delete(array, anomaly_indices)
    label = np.delete(label, anomaly_indices)
    return array, label


"""数据文件路径"""
file_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\剪裁后PKL数据"
file_name_1 = r"\CH4_0dB.pkl"
file_name_2 = r"\CH4_10dB.pkl"
file_name_3 = r"\CH4_20dB.pkl"
file_name_4 = r"\C2H2_0dB.pkl"
file_name_5 = r"\C2H2_10dB.pkl"
file_name_6 = r"\C2H2_20dB.pkl"

""""加载数据"""
"""
# CH4_label = read_pickle_to_array(file_path, file_name_1)
# CH4_label = CH4_label[:, :, np.newaxis]
# print("CH4_label.shape :", CH4_label.shape)
# print("============================================================================")
# CH4_10dB = read_pickle_to_array(file_path, file_name_2)
# CH4_10dB = CH4_10dB[:, np.newaxis, :]
# print("CH4_10dB.shape", CH4_10dB.shape)
# print("============================================================================")
# CH4_20dB = read_pickle_to_array(file_path, file_name_3)
# CH4_20dB = CH4_20dB[:, np.newaxis, :]
# print("CH4_20dB.shape", CH4_20dB.shape)
# print("============================================================================")
CH4_label = np.load(r"./correction_CH4_label.npy")
CH4_label = CH4_label[:, :, np.newaxis]
print("CH4_label.shape :", CH4_label.shape)  # (10000, 2000, 1)
print("============================================================================")

CH4_10dB = np.load(r"./correction_CH4_10dB.npy")
CH4_10dB = CH4_10dB[:, np.newaxis, :]
print("CH4_10dB.shape", CH4_10dB.shape)  # (10000, 1, 2000)
print("============================================================================")

CH4_20dB = np.load(r"./correction_CH4_20dB.npy")
CH4_20dB = CH4_20dB[:, np.newaxis, :]
print("CH4_20dB.shape", CH4_20dB.shape)   # (10000, 1, 2000)
print("============================================================================")
"""

# MLP数据格式
CH4_label = np.load(r"./correction_CH4_label.npy")
print("CH4_label.shape :", CH4_label.shape)  # (10000, 2000)

# loading input data
CH4_10dB = np.load(r"./correction_CH4_10dB.npy")

CH4_20dB = np.load(r"./correction_CH4_20dB.npy")
print("CH4_20dB.shape", CH4_20dB.shape)   # (10000, 2000)
CH4_input = CH4_20dB
print("CH4_input shape: ", CH4_input.shape)  # (20000, 2000)
print("============================================================================")
x_test = np.load(r"./mlp_x_test.npy")
y_test = np.load(r"./mlp_y_test.npy")

"""转为torch类型"""
CH4_label = torch.from_numpy(CH4_label)
CH4_label = CH4_label.type(torch.cuda.FloatTensor)

CH4_10dB = torch.from_numpy(CH4_10dB)
CH4_10dB = CH4_10dB.type(torch.cuda.FloatTensor)


CH4_20dB = torch.from_numpy(CH4_20dB)
CH4_20dB = CH4_20dB.type(torch.cuda.FloatTensor)

x_test = array_to_tensor(x_test)

print("============================================================================")

"""
sample1 = CH4_20dB[9999]
sample1 = sample1.reshape((1, 1, 2000))
sample2 = CH4_20dB[8000]
sample2 = sample2.reshape((1, 1, 2000))
sample3 = CH4_20dB[7000]
sample3 = sample3.reshape((1, 1, 2000))
sample4 = CH4_20dB[6000]
sample4 = sample4.reshape((1, 1, 2000))
CH4_label = np.squeeze(CH4_label.cpu().numpy())
sample5 = CH4_10dB[9999]
sample5 = sample5.reshape((1, 1, 2000))
sample6 = CH4_20dB[2000]
sample6 = sample6.reshape((1, 1, 2000))
"""
sample1 = CH4_20dB[9999]
sample1 = sample1.reshape((1, 2000))
sample2 = CH4_20dB[8000]
sample2 = sample2.reshape((1, 2000))
sample3 = CH4_20dB[7000]
sample3 = sample3.reshape((1, 2000))
sample4 = CH4_20dB[6000]
sample4 = sample4.reshape((1, 2000))
CH4_label = np.squeeze(CH4_label.cpu().numpy())
sample5 = CH4_10dB[9999]
sample5 = sample5.reshape((1, 2000))
sample6 = CH4_20dB[2000]
sample6 = sample6.reshape((1, 2000))

Gpu = torch.device("cuda")

x_test1 = x_test[100].reshape((1, 2000)).to(Gpu)
x_test2 = x_test[300].reshape((1, 2000)).to(Gpu)
x_test3 = x_test[500].reshape((1, 2000)).to(Gpu)
x_test4 = x_test[800].reshape((1, 2000)).to(Gpu)
# t = np.linspace(0, 1, 2000)
# """y = 7sin(2pi*200t) + 5sin(2pi*400t) + 3sin(2pi*600t)"""
# y = 7 * np.sin(2*np.pi*5*t) + 0.8*np.sin(2*np.pi*400*t) + 0.3*np.sin(2*np.pi*600*t)
# y = y.reshape((1, 1, 2000))


model = Filter5().to(Gpu)
# model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\滤波模型\filternet1.pt"
model_save_path = r"./透射谱模型7.pt"
model.load_state_dict(torch.load(model_save_path))
model.eval()

sample1 = sample1.to(Gpu)
restored1 = model(sample1)
sample2 = sample2.to(Gpu)
restored2 = model(sample2)
sample3 = sample3.to(Gpu)
restored3 = model(sample3)
sample4 = sample4.to(Gpu)
restored4 = model(sample4)
sample5 = sample5.to(Gpu)
restored5 = model(sample5)
restored6 = model(sample6)

y_test_predict1 = model(x_test1)
y_test_predict2 = model(x_test2)
y_test_predict3 = model(x_test3)
y_test_predict4 = model(x_test4)
# y = torch.from_numpy(y).type(torch.cuda.FloatTensor)
# y = y.to(Gpu)
# restored_y = model(y)
# y = np.squeeze(y.cpu().numpy())

restored1 = np.squeeze(restored1.cpu().detach().numpy())
restored2 = np.squeeze(restored2.cpu().detach().numpy())
restored3 = np.squeeze(restored3.cpu().detach().numpy())
restored4 = np.squeeze(restored4.cpu().detach().numpy())
restored5 = np.squeeze(restored5.cpu().detach().numpy())
restored6 = np.squeeze(restored6.cpu().detach().numpy())

restored1 = smoothing(restored1)
restored2 = smoothing(restored2)
restored3 = smoothing(restored3)
restored4 = smoothing(restored4)
restored5 = smoothing(restored5)
restored6 = smoothing(restored6)

# restored_y = np.squeeze(restored_y.cpu().detach().numpy())
y_test_predict1 = np.squeeze(y_test_predict1.cpu().detach().numpy())
y_test_predict2 = np.squeeze(y_test_predict2.cpu().detach().numpy())
y_test_predict3 = np.squeeze(y_test_predict3.cpu().detach().numpy())
y_test_predict4 = np.squeeze(y_test_predict4.cpu().detach().numpy())
# plt.figure()
# plt.plot(y)
# plt.plot(restored_y)

y_test_predict1, y_test1 = smoothing_n_comparision(y_test_predict1, y_test[100])
y_test_predict2, y_test2 = smoothing_n_comparision(y_test_predict2, y_test[300])
y_test_predict3, y_test3 = smoothing_n_comparision(y_test_predict3, y_test[500])
y_test_predict4, y_test4 = smoothing_n_comparision(y_test_predict4, y_test[800])

CH4_10dB = CH4_10dB.cpu().numpy()
CH4_10dB_sample = CH4_10dB[9999]
CH4_10dB_sample = np.squeeze(CH4_10dB_sample)

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title("10dB signal restoration")
# plt.plot(CH4_10dB_sample, color="lightcoral", label="10dB 1000ppm")
# plt.plot(restored5, color="cornflowerblue", label="restored 1000ppm")
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.title("10dB signal restoration")
# plt.plot(CH4_label[9999], color="lightcoral", label="low noise", linewidth=3)
# plt.plot(restored5, color="cornflowerblue", label="restored 1000ppm", linewidth=3)
# plt.legend()
plt.figure()
plt.title("original comparision")
plt.plot(np.squeeze(sample1.cpu().detach().numpy()))
plt.plot(np.squeeze(sample2.cpu().detach().numpy()))
plt.plot(np.squeeze(sample3.cpu().detach().numpy()))
plt.plot(np.squeeze(sample4.cpu().detach().numpy()))
plt.plot(np.squeeze(sample6.cpu().detach().numpy()))

plt.figure()
plt.title("restored comparision")
plt.plot(restored1, label="1000ppm")
plt.plot(restored2, label="800ppm")
plt.plot(restored3, label="700ppm")
plt.plot(restored4, label="600ppm")
plt.plot(restored6, label="200ppm")
plt.legend()

plt.figure()
plt.subplot(2, 2, 1)
plt.title("20dB 1000 ppm Comparision")
plt.plot(CH4_label[9999], label="low noise", color="lightcoral", linewidth=3)
plt.plot(restored1, label="restored", color="cornflowerblue")
plt.legend()
plt.subplot(2, 2, 2)
plt.title("20dB 200 ppm Comparision")
plt.plot(CH4_label[2000], label="low noise", color="lightcoral", linewidth=3)
plt.plot(restored6, label="restored", color="cornflowerblue")
plt.legend()
plt.subplot(2, 2, 3)
plt.title("20dB 700 ppm Comparision")
plt.plot(CH4_label[7000], label="low noise",color="lightcoral", linewidth=3)
plt.plot(restored3, label="restored", color="cornflowerblue")
plt.legend()
plt.subplot(2, 2, 4)
plt.title("20dB 600 ppm Comparision")
plt.plot(CH4_label[6000], label="low noise", color="lightcoral", linewidth=3)
plt.plot(restored4, label="restored", color="cornflowerblue")
plt.legend()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(y_test1, label="low noise", color="lightcoral", linewidth=3)
plt.plot(y_test_predict1, label="restored", color="cornflowerblue")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(y_test2, label="low noise", color="lightcoral", linewidth=3)
plt.plot(y_test_predict2, label="restored", color="cornflowerblue")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(y_test3, label="low noise", color="lightcoral", linewidth=3)
plt.plot(y_test_predict3, label="restored", color="cornflowerblue")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(y_test4, label="low noise", color="lightcoral", linewidth=3)
plt.plot(y_test_predict4, label="restored", color="cornflowerblue")
plt.legend()


plt.figure()
plt.plot(y_test_predict4, label="y_test4")
plt.plot(y_test_predict3, label="y_test3")
plt.plot(y_test_predict2, label="y_test2")
plt.plot(y_test_predict1, label="y_test1")
plt.legend()
plt.show()