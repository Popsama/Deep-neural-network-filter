# author: SaKuRa Pop
# data: 2020/11/16 14:28
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
from FilterNet_utils import ConvBNReLU, pooling1d, global_average_pooling1d, Filter, read_pickle_to_array, Data_set, training_1, \
    training_2, Filter2
import time
""""加载数据"""
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

"""转为torch类型"""
CH4_label = torch.from_numpy(CH4_label)
CH4_label = CH4_label.type(torch.cuda.FloatTensor)
print(CH4_label.type())
print("============================================================================")
CH4_10dB = torch.from_numpy(CH4_10dB)
CH4_10dB = CH4_10dB.type(torch.cuda.FloatTensor)
print(CH4_10dB.type())
print("============================================================================")
CH4_20dB = torch.from_numpy(CH4_20dB)
CH4_20dB = CH4_20dB.type(torch.cuda.FloatTensor)
print(CH4_20dB.type())
print("============================================================================")


"""生成模型实例"""
Gpu = torch.device("cuda")
# filter_net = Filter().to(Gpu)
filter_net = Filter2().to(Gpu)

"""定义criterion, optimizer"""
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter_net.parameters(), lr=0.01)

sample = CH4_20dB[9999]
sample = sample.reshape((1, 1, 2000))
label = CH4_label[9999]
label = label.reshape((1, 2000, 1))

loss_list = []
for e in range(1000):
    sample = sample.to(Gpu)
    label = label.to(Gpu)
    optimizer.zero_grad()
    prediction = filter_net(sample)
    loss = criterion(prediction, label)
    loss_list.append(float(loss))
    loss.backward()
    optimizer.step()
    print(loss.item())


prediction = filter_net(sample)
prediction = np.squeeze(prediction.cpu().detach().numpy())
label = np.squeeze(label.cpu().numpy())

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(loss_list)
plt.subplot(2, 1, 2)
ax = plt.gca()
ax.set_yscale("log")
plt.plot(loss_list)

plt.figure()
plt.plot(label)
plt.plot(prediction)
plt.show()

