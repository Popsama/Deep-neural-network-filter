# author: SaKuRa Pop
# data: 2021/1/9 21:30
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
from FilterNet_utils import array_to_tensor, Filter, Filter2, Filter3, read_pickle_to_array, Data_set, training_1, \
    training_2, Filter4, Filter5
import time
from sklearn.model_selection import train_test_split

""""加载数据"""
"""
# loading labels
CH4_label = np.load(r"./correction_CH4_label.npy")
# CH4_label = np.concatenate((CH4_label, CH4_label), axis=0)
CH4_label = CH4_label[:, :, np.newaxis]
print("CH4_label.shape :", CH4_label.shape)  # (10000 * 2, 2000, 1)
print("============================================================================")

# loading input data
# CH4_10dB = np.load(r"./correction_CH4_10dB.npy")
# print("CH4_10dB.shape", CH4_10dB.shape)  # (10000, 2000)
# print("============================================================================")
CH4_20dB = np.load(r"./correction_CH4_20dB.npy")
print("CH4_20dB.shape", CH4_20dB.shape)   # (10000, 2000)
print("============================================================================")
# concatenate 10dB & 20 dB data together as input data
# CH4_input = np.concatenate((CH4_10dB, CH4_20dB), axis=0)
CH4_input = CH4_20dB[:, np.newaxis, :]
print("CH4_input shape: ", CH4_input.shape)  # (20000, 1, 2000)
print("============================================================================")
"""

# MLP数据格式
CH4_label = np.load(r"./correction_CH4_label.npy")
# CH4_label = np.concatenate((CH4_label, CH4_label), axis=0)
print("CH4_label.shape :", CH4_label.shape)  # (10000, 2000)

# loading input data
CH4_20dB = np.load(r"./correction_CH4_20dB.npy")
print("CH4_20dB.shape", CH4_20dB.shape)   # (10000, 2000)
# concatenate 10dB & 20 dB data together as input data
# CH4_input = np.concatenate((CH4_10dB, CH4_20dB), axis=0)
CH4_input = CH4_20dB
print("CH4_input shape: ", CH4_input.shape)  # (20000, 2000)
print("============================================================================")


# 数据分割
x_train, x_test, y_train, y_test = train_test_split(CH4_input, CH4_label, test_size=0.1, random_state=2)
np.save(r"./mlp_x_train.npy", x_train)
np.save(r"./mlp_x_test.npy", x_test)
np.save(r"./mlp_y_train.npy", y_train)
np.save(r"./mlp_y_test.npy", y_test)

# """转为torch类型"""
x_train = array_to_tensor(x_train)
x_test = array_to_tensor(x_test)
y_train = array_to_tensor(y_train)
y_test = array_to_tensor(y_test)


CH4_label = array_to_tensor(CH4_label)
CH4_input = array_to_tensor(CH4_input)

# CH4_10dB = torch.from_numpy(CH4_10dB)
# CH4_10dB = CH4_10dB.type(torch.cuda.FloatTensor)
# print(CH4_10dB.type())  # torch.cuda.FloatTensor
# print("============================================================================")
# CH4_20dB = torch.from_numpy(CH4_20dB)
# CH4_20dB = CH4_20dB.type(torch.cuda.FloatTensor)
# print(CH4_20dB.type())  # torch.cuda.FloatTensor
# print("============================================================================")


"""准备数据集"""
# data_set = Data_set(CH4_input, CH4_label, 100)
data_set = Data_set(x_train, y_train, 100)


"""生成模型实例"""
Gpu = torch.device("cuda")
# filter_net = Filter().to(Gpu)
filter_net = Filter5().to(Gpu)  # which Filter to train
"""定义criterion, optimizer"""
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter_net.parameters(), lr=0.0001)

"""训练模型"""
begin_time = time.time()
# 用training 1 带 test error
# train_loss, test_loss = training_1(filter_net, data_set, x_test, y_test, Gpu, optimizer, criterion, epochs=1000,
#                                    plot_ch4=CH4_20dB, plot_label=CH4_label, adjust=False, plot=True)

# 用training 2 不带test error
train_loss = training_2(filter_net, data_set, Gpu, optimizer, criterion, epochs=2000,
                        plot_ch4=CH4_20dB, plot_label=CH4_label, adjust=False, plot=True)

end_time = time.time()
total_time_cost = (end_time - begin_time) / 60
print("总训练用时：{} 分钟".format(total_time_cost))
print("总时长：{} 小时".format(total_time_cost/60.0))

"""保存模型"""
model_save_path = r"./透射谱模型7.pt"
torch.save(filter_net.state_dict(), model_save_path)
print("saved finished ")
# 本次训练10,000epochs， lr= 0.001 remember this
# 透射谱模型4.pt 1000 epochs training 1, Filter 2, lr=0.1
# 透射谱模型4.pt 1000 epochs training 2, Filter 2, lr=0.01
# 透射谱模型4.pt 1000 epochs training 2, Filter 1, lr=0.01
# 透射谱模型4.pt 2000 epochs training 2, Filter 2改, lr=0.001
# 透射谱模型5.pt 2000 epochs training 2, Filter 2改改, lr=0.001
# 透射谱模型5.pt 2000 epochs training 2, Filter 4, lr=0.0001  dropout rate = 0.2
# 透射谱模型6.pt 2000 epochs training 2, Filter 5, lr=0.001  loss =0.0044
# 透射谱模型7.pt 2000 epochs training 2, Filter 5, lr=0.0001  loss =0.0024

sample1 = CH4_input[9999]
sample1 = sample1.reshape((1, 1, 2000))
sample2 = CH4_input[8999]
sample2 = sample2.reshape((1, 1, 2000))
CH4_input = np.squeeze(CH4_input.cpu().numpy())
CH4_label = np.squeeze(CH4_label.cpu().numpy())


# with torch.no_grad():
# filter_net.eval() # 这句还是尽量避免 不要使用 ？
filter_net = filter_net.to(Gpu)
sample1 = sample1.to(Gpu)
restored1 = filter_net(sample1)
sample2 = sample2.to(Gpu)
restored2 = filter_net(sample2)
restored1 = np.squeeze(restored1.cpu().detach().numpy())
restored2 = np.squeeze(restored2.cpu().detach().numpy())
print("restored sample shape: ", restored1.shape)

"""可视化训练"""
plt.figure()
plt.subplot(2, 1, 1)
ax = plt.gca()
ax.tick_params(direction="in")
plt.plot(train_loss, label="train loss")

plt.subplot(2, 1, 2)
plt.plot(train_loss, label="train loss")
ax = plt.gca()
ax.tick_params(direction="in")
ax.set_yscale("log")

plt.figure()
plt.subplot(1, 3, 1)
ax = plt.gca()
ax.tick_params(direction="in")
plt.title("Noisy")
plt.plot(CH4_20dB[9999])

plt.subplot(1, 3, 2)
ax = plt.gca()
ax.tick_params(direction="in")
plt.title("Low noise")
plt.plot(CH4_label[9999])

plt.subplot(1, 3, 3)
ax = plt.gca()
ax.tick_params(direction="in")
plt.title("Restored/Filtered")
plt.plot(restored1)

plt.figure()
plt.subplot(2, 2, 1)
ax = plt.gca()
ax.tick_params(direction="in")
plt.title("Comparision")
plt.plot(CH4_label[9999], label="low noise")
plt.plot(restored1, label="restored")
plt.legend()

plt.subplot(2, 2, 2)
ax = plt.gca()
ax.tick_params(direction="in")
plt.title("Comparision")
plt.plot(CH4_label[8000], label="low noise")
plt.plot(restored2, label="restored")
plt.legend()
plt.show()
