# author: SaKuRa Pop
# data: 2020/11/8 11:30
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
from FilterNet_utils import array_to_tensor, Filter, read_pickle_to_array, Data_set, training_1, training_2
import time
from sklearn.model_selection import train_test_split

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
CH4_label = CH4_label[:, :, np.newaxis]
print("CH4_label.shape :", CH4_label.shape)  # (10000, 2000, 1)
print("============================================================================")
CH4_10dB = read_pickle_to_array(file_path, file_name_2)

CH4_10dB = CH4_10dB[:, np.newaxis, :]
print("CH4_10dB.shape", CH4_10dB.shape)  # (10000, 1, 2000)
print("============================================================================")
CH4_20dB = read_pickle_to_array(file_path, file_name_3)
CH4_20dB = CH4_20dB[:, np.newaxis, :]
print("CH4_20dB.shape", CH4_20dB.shape)   # (10000, 1, 2000)
print("============================================================================")

# C2H2_label = read_pickle_to_array(file_path, file_name_4)
# C2H2_label = C2H2_label[:, :, np.newaxis]
# print("CH4_label.shape", CH4_label.shape)
# print("============================================================================")
# C2H2_10dB = read_pickle_to_array(file_path, file_name_5)
# C2H2_10dB = C2H2_10dB[:, np.newaxis, :]
# print("C2H2_10dB.shap", C2H2_10dB.shape)
# print("============================================================================")
# C2H2_20dB = read_pickle_to_array(file_path, file_name_6)
# C2H2_20dB = C2H2_20dB[:, np.newaxis, :]
# print("C2H2_20dB.shape", C2H2_20dB.shape)
# print("============================================================================")

# x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(CH4_20dB, CH4_label, test_size=0.1, random_state=2)
# x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(CH4_10dB, CH4_label, test_size=0.1, random_state=3)
# """转为torch类型"""
# x_train_1 = array_to_tensor(x_train_1)
# x_test_1 = array_to_tensor(x_test_1)
# y_train_1 = array_to_tensor(y_train_1)
# y_test_1 = array_to_tensor(y_test_1)
# x_train_2 = array_to_tensor(x_train_2)
# x_test_2 = array_to_tensor(x_test_2)
# y_train_2 = array_to_tensor(y_train_2)
# y_test_2 = array_to_tensor(y_test_2)

CH4_label = torch.from_numpy(CH4_label)
CH4_label = CH4_label.type(torch.cuda.FloatTensor)
print(CH4_label.type())  # torch.cuda.FloatTensor
print("============================================================================")
CH4_10dB = torch.from_numpy(CH4_10dB)
CH4_10dB = CH4_10dB.type(torch.cuda.FloatTensor)
print(CH4_10dB.type())  # torch.cuda.FloatTensor
print("============================================================================")
CH4_20dB = torch.from_numpy(CH4_20dB)
CH4_20dB = CH4_20dB.type(torch.cuda.FloatTensor)
print(CH4_20dB.type())  # torch.cuda.FloatTensor
print("============================================================================")

# CH4_20dB = CH4_20dB / 20
# CH4_label = CH4_label / 20

"""准备数据集"""
data_set = Data_set(CH4_20dB, CH4_label, 100)
# data_set_1 = Data_set(x_train_1, y_train_1, 100)
# data_set_2 = Data_set(x_train_2, y_train_2, 100)

"""生成模型实例"""
Gpu = torch.device("cuda")
filter_net = Filter().to(Gpu)

"""定义criterion, optimizer"""
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter_net.parameters(), lr=0.01)

"""训练模型"""
begin_time = time.time()
# train_loss, test_loss = training_1(filter_net, data_set, x_test_1, y_test_1, Gpu, optimizer, criterion, epochs=300,
#                                    plot_ch4=CH4_20dB, plot_label=CH4_label, adjust=True, plot=True)
train_loss = training_2(filter_net, data_set, Gpu, optimizer, criterion, epochs=10000,
                        plot_ch4=CH4_20dB, plot_label=CH4_label, adjust=False, plot=True)
end_time = time.time()
total_time_cost = (end_time - begin_time) / 60
print("总训练用时：{} 分钟".format(total_time_cost))

"""保存模型"""
model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\滤波模型\filternet3.pt"
torch.save(filter_net.state_dict(), model_save_path)


sample1 = CH4_20dB[9999]
sample1 = sample1.reshape((1, 1, 2000))
sample2 = CH4_20dB[8000]
sample2 = sample2.reshape((1, 1, 2000))
CH4_20dB = np.squeeze(CH4_20dB.cpu().numpy())
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
plt.plot(train_loss, label="train loss")


plt.subplot(2, 1, 2)
plt.plot(train_loss, label="train loss")

ax = plt.gca()
ax.set_yscale("log")

plt.figure()
plt.subplot(1, 3, 1)
plt.title("Noisy")
plt.plot(CH4_20dB[9999])

plt.subplot(1, 3, 2)
plt.title("Low noise")
plt.plot(CH4_label[9999])

plt.subplot(1, 3, 3)
plt.title("Restored/Filtered")
plt.plot(restored1)

plt.figure()

plt.subplot(2, 2, 1)
plt.title("Comparision")
plt.plot(CH4_label[9999], label="low noise")
plt.plot(restored1, label="restored")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Comparision")
plt.plot(CH4_label[8000], label="low noise")
plt.plot(restored2, label="restored")
plt.legend()
plt.show()
