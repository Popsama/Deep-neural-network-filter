import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torch
import pickle
import torch.utils.data as Data
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


def ConvBNReLU(in_channels, out_channels, kernel_size, stride):
    """
    自定义包括BN ReLU的卷积层，输入(N,in_channels,in_Length)
    输出(N, out_channels, out_Length)，卷积后进行批归一化，
    然后进行RELU激活。
    :param in_channels: 输入张量的通道数
    :param out_channels: 输出张量的通道数
    :param kernel_size: 卷积核尺寸
    :param stride: 卷积核滑动步长
    :return: BN RELU后的卷积输出
    """
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


def pooling1d(input_tensor, kernel_size=3, stride=2):
    """
    最大池化函数，输入形如(N, Channels, Length),
    输出形如(N, Channels, Length)的功能。
    :param input_tensor:要被最大池化的输入张量
    :param kernel_size:池化尺寸。在多大尺寸内进行最大池化操作
    :param stride:池化层滑动补偿
    :return:池化后输出张量
    """
    result = F.max_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
    return result


def average_pool1d(input_tensor, kernel_size=3, stride=2):
    result = F.avg_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
    return result


def global_average_pooling1d(input_tensor, output_size=1):
    """
    全局平均池化函数，将length压缩成output_size。
    输入(N, C, Input_size)
    输出(N, C, output_size)
    :param input_tensor: 输入张量
    :param output_size: 输出张量
    :return:全剧平均池化输出
    """
    result = F.adaptive_max_pool1d(input_tensor, output_size)
    return result


class Filter(nn.Module):
    """
    神经网络滤波器
    input -     Conv1 -      Maxpool -  Conv2 -      Maxpool -    Conv3 -     Maxpool  -  Conv4 -     Conv5 - Globaverg
    (1, 2000) - (10, 999) -  (10, 498)- (100, 248) - (100, 122) - (500, 60) - (500, 29) - (1000, 14)-(2000, 6)-(2000,1)
    """
    def __init__(self):
        super(Filter, self).__init__()
        self.conv1 = ConvBNReLU(in_channels=1, out_channels=10, kernel_size=7, stride=2)
        self.conv2 = ConvBNReLU(in_channels=10, out_channels=100, kernel_size=7, stride=2)
        self.conv3 = ConvBNReLU(in_channels=100, out_channels=500, kernel_size=7, stride=2)
        self.conv4 = ConvBNReLU(in_channels=500, out_channels=1000, kernel_size=7, stride=2)
        self.conv5 = ConvBNReLU(in_channels=1000, out_channels=2000, kernel_size=7, stride=2)
        self.globavgpool1 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x = pooling1d(self.conv1(x))
        x = pooling1d(self.conv2(x))
        x = pooling1d(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.globavgpool1(x)
        return out


class Filter2(nn.Module):
    """
    神经网络滤波器
    input -     Conv1 -      Maxpool -  Conv2 -      Maxpool -    Conv3 -     Maxpool  -  Conv4 -     Conv5 - Globaverg
    (1, 2000) - (10, 997) -  (10, 498)- (100, 248) - (100, 122) - (500, 60) - (500, 29) - (1000, 14)-(2000, 6)-(2000,1)
    """
    def __init__(self):
        super(Filter2, self).__init__()
        self.conv1 = ConvBNReLU(in_channels=1, out_channels=10, kernel_size=32, stride=2)
        self.maxpool1 = nn.AvgPool1d(kernel_size=10, stride=2)
        self.conv2 = ConvBNReLU(in_channels=10, out_channels=50, kernel_size=16, stride=2)
        self.maxpool2 = nn.AvgPool1d(kernel_size=5, stride=2)
        self.conv3 = ConvBNReLU(in_channels=50, out_channels=100, kernel_size=8, stride=2)
        self.maxpool3 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=117*50, out_features=2000)
        # self.fc2 = nn.Linear(in_features=4000, out_features=2000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, self.flatten_features(x))
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        x = x.unsqueeze(-1)
        return x

    @staticmethod
    def flatten_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Filter3(nn.Module):
    """
    神经网络滤波器
    input -     Conv1 -      Maxpool -  Conv2 -      Maxpool -    Conv3 -     Maxpool  -  Conv4 -     Conv5 - Globaverg
    (1, 2000) - (10, 997) -  (10, 498)- (100, 248) - (100, 122) - (500, 60) - (500, 29) - (1000, 14)-(2000, 6)-(2000,1)
    """
    def __init__(self):
        super(Filter3, self).__init__()
        self.conv1 = ConvBNReLU(in_channels=1, out_channels=10, kernel_size=32, stride=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=10, stride=2)
        self.conv2 = ConvBNReLU(in_channels=10, out_channels=50, kernel_size=16, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.conv3 = ConvBNReLU(in_channels=50, out_channels=100, kernel_size=8, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=117*50, out_features=2000)
        # self.fc2 = nn.Linear(in_features=4000, out_features=2000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, self.flatten_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = x.unsqueeze(-1)
        return x

    @staticmethod
    def flatten_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Filter4(nn.Module):  # 否决了。全连接就算只有四层参数也太多了
    """
    神经网络滤波器
    input -     Conv1 -      Maxpool -  Conv2 -      Maxpool -    Conv3 -     Maxpool  -  Conv4 -     Conv5 - Globaverg
    (1, 2000) - (10, 997) -  (10, 498)- (100, 248) - (100, 122) - (500, 60) - (500, 29) - (1000, 14)-(2000, 6)-(2000,1)
    """
    def __init__(self):
        super(Filter4, self).__init__()
        self.fc1 = nn.Linear(in_features=2000, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=500, out_features=2000)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

    @staticmethod
    def flatten_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Filter5(nn.Module):  # 三层 hidden layer 效果好像很不错的亚子  2000-500-100-2000
    """
    神经网络滤波器
    input -     Conv1 -      Maxpool -  Conv2 -      Maxpool -    Conv3 -     Maxpool  -  Conv4 -     Conv5 - Globaverg
    (1, 2000) - (10, 997) -  (10, 498)- (100, 248) - (100, 122) - (500, 60) - (500, 29) - (1000, 14)-(2000, 6)-(2000,1)
    """
    def __init__(self):
        super(Filter5, self).__init__()
        self.fc1 = nn.Linear(in_features=2000, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=100, out_features=2000)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x


def read_pickle_to_array(path, name):
    """
    读取二进制文件并创建为np array类型
    :param path: 读取文件的路径
    :param name: 读取文件的文件名
    :return: ndarray类型的数据
    """
    file = open(path + name, "rb")
    array = pickle.load(file)
    array = np.array(array)
    return array


def Data_set(input_data, label_data, batch_size):
    """
    生成data_loader实例。可以定义batch_size
    :param input_data: 希望作为训练input的数据，tensor类型
    :param label_data: 希望作为训练label的数据，tensor类型
    :param batch_size: batch size
    :return: data_loader实例
    """
    data_set = Data.TensorDataset(input_data, label_data)
    data_loader = Data.DataLoader(dataset=data_set,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=0)
    return data_loader


def training_1(model, train_loader, test_x, test_y, device, optimizer, criterion, epochs, plot_ch4, plot_label,
               adjust=False, plot=False):
    """
    训练模型，输入模型，数据集，GPU设备，选择的优化器以及损失函数，在设置的epoch内进行模型优化。
    adjust开启时将根据epoch自适应的调整learning rate。
    只有当adjust开启时，plot才能开启。否则plot功能永远关闭。
    plot开启时，将绘制输入的plot_ch4以及plot_label。每一代之后根据更新优化的参数，模型计算plot_ch4，并绘制输出与plot_label进行
    对比
    与training 2不用，1需要输入测试集，可以计算测试误差。
    :param model: 输入的训练模型, untrained model
    :param train_loader: 输入的训练数据集
    :param test_x: using for compute test error
    :param test_y: same as above
    :param device: GPU or  cpu
    :param optimizer: the chosen optimizer
    :param criterion: the loss function
    :param epochs: iteration running on models
    :param plot_ch4: a sample chosen from dataset to be computed and plotted
    :param plot_label: a sample label chosen from dataset to be plotted
    :param adjust: adaptive learning rate along with epochs when switch to True
    :param plot: only adjust = True will switch on.
    :return: trained loss & test loss
    """
    model.train()
    model.to(device)
    iteration_loss_list = []
    test_error_list = []
    if adjust is False:
        for e in range(epochs):
            for index, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                optimizer.zero_grad()
                prediction1 = model(batch_x)
                prediction2 = model(test_x)
                loss = criterion(prediction1, batch_y)
                iteration_loss_list.append(float(loss))
                test_error = criterion(prediction2, test_y)
                test_error_list.append(float(test_error))
                loss.backward()
                optimizer.step()
                print("epoch: {} [{}/{} {:.2f}%] train loss: {}  test loss: {}".format(e, index*len(batch_x),
                                                                                       len(train_loader.dataset),
                                                                                       100*index/len(train_loader),
                                                                                       loss.item(), test_error.item())
                      )
            # epoch_loss_list.append(loss)

    elif adjust is True:
        schuduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        if plot is True:
            plt.figure()
            plt.ion()
            plt.show()
            for e in range(epochs):
                for index, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    optimizer.zero_grad()
                    prediction1 = model(batch_x)
                    prediction2 = model(test_x)
                    loss = criterion(prediction1, batch_y)
                    iteration_loss_list.append(float(loss))
                    test_error = criterion(prediction2, test_y)
                    test_error_list.append(float(test_error))
                    loss.backward()
                    optimizer.step()
                    schuduler.step()
                    print("epoch: {} [{}/{} {:.2f}%] train loss: {}  test loss: {}".format(e, index * len(batch_x),
                                                                                           len(train_loader.dataset),
                                                                                           100 * index / len(
                                                                                               train_loader),
                                                                                           loss.item(), test_error.item())
                          )
                sample = plot_ch4[9999]
                sample = sample.reshape((1, 1, 2000))
                sample = array_to_tensor(sample)
                plot_prediction = model(sample)
                plot_prediction = np.squeeze(plot_prediction.cpu().detach().numpy())
                CH4_label = np.squeeze(plot_label)
                plt.cla()
                plt.plot(CH4_label[9999])
                plt.plot(plot_prediction)
                plt.pause(0.1)
            plt.ioff()
            plt.show()
    return iteration_loss_list, test_error_list  # , epoch_loss_list


def training_2(model, train_loader, device, optimizer, criterion, epochs, plot_ch4, plot_label,
               adjust=False, plot=False):
    """
    训练模型，输入模型，数据集，GPU设备，选择的优化器以及损失函数，在设置的epoch内进行模型优化。
    adjust开启时将根据epoch自适应的调整learning rate。
    只有当adjust开启时，plot才能开启。否则plot功能永远关闭。
    plot开启时，将绘制输入的plot_ch4以及plot_label。每一代之后根据更新优化的参数，模型计算plot_ch4，并绘制输出与plot_label进行
    对比
    :param model: 输入的训练模型, untrained model
    :param train_loader: 输入的训练数据集
    :param test_x: using for compute test error
    :param optimizer: the chosen optimizer
    :param criterion: the loss function
    :param epochs: iteration running on models
    :param plot_ch4: a sample chosen from dataset to be computed and plotted
    :param plot_label: a sample label chosen from dataset to be plotted
    :param adjust: adaptive learning rate along with epochs when switch to True
    :param plot: only adjust = True will switch on.
    :return: trained loss & test loss
    """
    model.train()
    model.to(device)
    iteration_loss_list = []
    if adjust is False:
        for e in range(epochs):
            for index, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                prediction1 = model(batch_x)
                loss = criterion(prediction1, batch_y)
                iteration_loss_list.append(float(loss))
                loss.backward()
                optimizer.step()
                print("epoch: {} [{}/{} {:.2f}%] train loss: {}".format(e, index*len(batch_x),
                                                                        len(train_loader.dataset),
                                                                        100*index/len(train_loader),
                                                                        loss.item())
                      )
            # epoch_loss_list.append(loss)

    elif adjust is True:
        schuduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        if plot is True:
            plt.figure()
            plt.ion()
            plt.show()
            for e in range(epochs):
                for index, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    prediction1 = model(batch_x)
                    loss = criterion(prediction1, batch_y)
                    iteration_loss_list.append(float(loss))
                    loss.backward()
                    optimizer.step()
                    schuduler.step()
                    print("epoch: {} [{}/{} {:.2f}%] train loss: {}".format(e, index * len(batch_x),
                                                                            len(train_loader.dataset),
                                                                            100 * index / len(train_loader),
                                                                            loss.item())
                          )
                sample = plot_ch4[9999]
                sample = sample.reshape((1, 1, 2000))
                plot_prediction = model(sample)
                plot_prediction = np.squeeze(plot_prediction.cpu().detach().numpy())
                CH4_label = np.squeeze(plot_label.cpu().numpy())
                plt.cla()
                plt.plot(CH4_label[9999])
                plt.plot(plot_prediction)
                plt.pause(0.1)
            plt.ioff()
            plt.show()
    return iteration_loss_list  # , epoch_loss_list


if __name__ == "__main__":

    Gpu = torch.device("cuda")
    Cpu = torch.device("cpu")
    filter_net = Filter4().to(Gpu)  # 模型加载到GPU上
    print(filter_net)                 # torch的print(model)只能打印层的细节，不包括每层的输出维度 有点遗憾
    summary(filter_net, (1, 2000))  # summary()很像keras的model.summary()
    # x = torch.randn((2, 2000))
    # y = filter_net(x)
    # print(y.shape)