# author: SaKuRa Pop
# data: 2021/4/17 14:40
import numpy as np
import torch
from scipy.signal import savgol_filter
from bp_kalman_filter import BpFilter, array_to_tensor
import matplotlib.pyplot as plt

"""模拟数据 （透射谱）"""
no_noise_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH_nonoise_spectral.npy"
CH4_no_noise_spectral = np.load(no_noise_path)  # (1000, 1111) 透射谱（吸收谱）
CH4_no_noise_spectral = CH4_no_noise_spectral.astype(np.float64)  # np.float64
noisy_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH_noisy_spectral.npy"
CH4_noisy_spectral = np.load(noisy_path)  # (1000, 1111) 透射谱（吸收谱）
CH4_noisy_spectral = CH4_noisy_spectral.astype(np.float64)  # np.float64
After_KF_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_afterKF.npy"
CH4_KF = np.load(After_KF_path)
CH4_KF = CH4_KF.astype(np.float64)

Gpu = torch.device("cuda")
Bp_kalman = BpFilter().to(Gpu)
model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\滤波模型\透射谱的滤波模型\BP_KF.pt"
Bp_kalman.load_state_dict(torch.load(model_save_path))

index = 600
test = CH4_KF[index]
print(test.shape)
SG_filtered = savgol_filter(test, 11, 2)

test = array_to_tensor(test)
test = test.reshape(1111, 1)
prediction = Bp_kalman(test)
prediction = prediction.cpu().detach().numpy()
test = test.cpu().detach().numpy()

plt.figure()
plt.plot(CH4_noisy_spectral[index], label="without any filtering")
plt.plot(test, alpha=1, label="only KF")
plt.plot(prediction, alpha=1, label="KF+ BP")
plt.plot(CH4_no_noise_spectral[index], linewidth=4, label="original signal")
plt.legend()


plt.figure()
plt.plot(SG_filtered, label="S-G filter")
plt.plot(prediction, alpha=0.8, label="KF+ BP")
plt.plot(CH4_no_noise_spectral[index], linewidth=4, label="original signal")
plt.legend()

plt.show()

