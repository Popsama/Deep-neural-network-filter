# author: SaKuRa Pop
# data: 2020/11/18 9:23
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
from FilterNet_utils import ConvBNReLU, pooling1d, global_average_pooling1d, Filter, read_pickle_to_array, Data_set, training_1, \
    training_2, array_to_tensor
from sklearn import linear_model

file_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\剪裁后PKL数据"
file_name_1 = r"\CH4_0dB.pkl"
CH4_label = read_pickle_to_array(file_path, file_name_1)
label_sample = CH4_label[5000]


CH4_500_experimental_file = open(r"D:\PYHTON\python3.7\DeepLearningProgram\TraceGasSensoring\常老师的意见\500ppm_CH4_intensity.pkl", "rb")
CH4_500_experimental = np.array(pickle.load(CH4_500_experimental_file))
CH4_500_experimental = CH4_500_experimental[:, np.newaxis, :]
CH4_500_experimental = CH4_500_experimental[:, :, 1000: 3000] * 4.8
print("dataset shape: ", CH4_500_experimental.shape)

sample = CH4_500_experimental[0]
sample = sample.reshape(1, 1, 2000)
print("sample shape : ", sample.shape)
sample = array_to_tensor(sample)

Gpu = torch.device("cuda")
model = Filter().to(Gpu)
model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\滤波模型\filternet1.pt"
model.load_state_dict(torch.load(model_save_path))

sample = sample.to(Gpu)
restored = model(sample)
restored = np.squeeze(restored.cpu().detach().numpy())

plt.figure()
plt.subplot(1, 2, 1)
plt.title("experimental 500ppm unknown SNR")
plt.plot(np.squeeze(CH4_500_experimental[0]), label="origin noisy signal")
plt.subplot(1, 2, 2)
plt.plot(restored, label="restored high SNR signal")
# plt.plot(label_sample, label="no noise signal")
plt.show()
