# author: SaKuRa Pop
# data: 2021/4/12 21:21
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Spectral_KF import SpectralKalmanFilter
from torch.utils.data import random_split
from FilterNet_utils import Data_set


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


class BpFilter(nn.Module):  # four hidden layer; seven hidden units in each layer (according to the original paper)
    """
    神经网络滤波器
    input -  layer 1 -  layer2 - layer3 - layer4 - output
    (1, 1) - (1, 7) -  (1, 7) - (1, 7) - (1, 7) - (1, 1)
    """
    def __init__(self):
        super(BpFilter, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=7)
        self.fc2 = nn.Linear(in_features=7, out_features=7)
        self.fc3 = nn.Linear(in_features=7, out_features=7)
        self.fc4 = nn.Linear(in_features=7, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x


def split_dataset(data, label, train_batch_size, train_size, validation_size):
    torch.manual_seed(0)  # 想复现的话可以设置一个随机种子
    data_set = Data.TensorDataset(data, label)
    train_size = int(len(data_set) * train_size)
    validate_size = int(len(data_set) * validation_size)
    test_size = len(data_set) - validate_size - train_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset=data_set,
                                                                                  lengths=[train_size,
                                                                                           validate_size,
                                                                                           test_size])
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   shuffle=True,
                                   batch_size=train_batch_size,
                                   num_workers=0)
    validation_loader = Data.DataLoader(dataset=validate_dataset,
                                        shuffle=True,
                                        batch_size=validate_size,
                                        num_workers=0)
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  shuffle=True,
                                  batch_size=test_size,
                                  num_workers=0)
    return train_loader, validation_loader, test_loader


def Fit(model, train_loader,  device, optimizer, criterion, epochs):
    """
    训练模型，输入模型，数据集，GPU设备，选择的优化器以及损失函数，在设置的epoch内进行模型优化。
    :param model: 输入的训练模型, untrained model
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param device: GPU or  cpu
    :param optimizer: the chosen optimizer
    :param criterion: the loss function
    :param epochs: iteration running on models
    :return: trained loss & test loss
    """
    model.train()
    model.to(device)
    iteration_loss_list = []
    validation_error_list = []
    for e in range(epochs):
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction1 = model(batch_x)
            loss = criterion(prediction1, batch_y)
            iteration_loss_list.append(float(loss))
            loss.backward()
            optimizer.step()
            print("epoch: {} [{}/{} {:.2f}%] train loss: {} ".format(e, i*len(batch_x),
                                                                     len(train_loader.dataset),
                                                                     100*i/len(train_loader),
                                                                     loss.item())
                  )
    return iteration_loss_list, validation_error_list


def plain_kalman(index):
    process_var = 0.009858427**2  # process noise variance
    sensor_var = 0.009223176**2  # measurement noise variance
    Z = CH4_noisy_spectral[index]
    nk = np.arange(0, 1111)
    Z = np.vstack((Z, nk))  # (2, 1111) measurement matrix
    U = CH4_no_noise_spectral[index]
    ones = np.ones((1111,))
    U = np.vstack((U, ones))  # (2, 1111) control matrix
    KF_Filter = SpectralKalmanFilter(U, Z, process_var, sensor_var)  # KF类的实例
    CH4_KF = KF_Filter.filtering()
    CH4_KF = CH4_KF[0]  # (1111,)
    return CH4_KF


"""experimental data"""
save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\实验数据\实验数据.npy"
CH4_experimental_data = np.load(save_path)
CH4_experimental_label = np.arange(10, 1010, 10)  # 100, 1  10ppm~1000ppm, step = 10ppm

"""模拟数据 （透射谱）"""
no_noise_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH_nonoise_spectral.npy"
CH4_no_noise_spectral = np.load(no_noise_path)  # (1000, 1111) 透射谱（吸收谱）
noisy_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH_noisy_spectral.npy"
CH4_noisy_spectral = np.load(noisy_path)  # (1000, 1111) 透射谱（吸收谱）

"""plain KF滤波1次"""
CH4_KF = np.zeros_like(CH4_no_noise_spectral)  # (1000, 1111)
for i in range(1000):
    CH4_KF[i] = plain_kalman(index=i)
    print(i)
print("CH4_KF shape: ", CH4_KF.shape)

"""KF滤波之后用BP再优化一次"""
input_data = CH4_KF.reshape(-1, 1)  # (1111000, 1) 将它作为BpKF的input
label_data = CH4_no_noise_spectral.reshape(-1, 1)  # label (1111000, 1)
input_data = array_to_tensor(input_data)  # torch.cuda.FloatTensor size(1111, 1)
label_data = array_to_tensor(label_data)  # torch.cuda.FloatTensor size(1111, 1)
print("input data shape:", input_data.shape)
print("label shape:", label_data.shape)


train_data, test_data, train_label, test_label = train_test_split(input_data, label_data, test_size=0.2, random_state=2)
train_loader = Data_set(train_data, train_label, 1111)

"""BP模型实例"""
Gpu = torch.device("cuda")
Bp_kalman = BpFilter().to(Gpu)

# for m in Bp_kalman.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_normal_(m.weight.data)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight.data)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Bp_kalman.parameters(), lr=0.01)
train_loss, validation_loss = Fit(Bp_kalman, train_loader, Gpu, optimizer, criterion, 50)
plt.figure()
plt.plot(train_loss, label="train loss")
plt.legend()
plt.yscale("log")
plt.show()


test = CH4_KF[600]
test = array_to_tensor(test)
test = test.reshape(1111, 1)
prediction = Bp_kalman(test)
prediction = prediction.cpu().detach().numpy()
test = test.cpu().detach().numpy()
plt.figure()
plt.plot(test, alpha=1, label="only KF")
plt.plot(prediction, alpha=1, label="KF+ BP")
plt.plot(CH4_no_noise_spectral[600], linewidth=4, label="original signal")
plt.legend()
plt.show()

"""保存模型"""
model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\滤波模型\透射谱的滤波模型\BP_KF.pt"
torch.save(Bp_kalman.state_dict(), model_save_path)
