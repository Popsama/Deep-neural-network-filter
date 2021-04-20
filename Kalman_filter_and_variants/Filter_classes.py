# author: SaKuRa Pop
# data: 2021/4/20 17:01
import torch
import matplotlib.pyplot as plt
from bp_kalman_filter import BpFilter, array_to_tensor, plain_kalman
from scipy.signal import savgol_filter
import pywt
import numpy as np


def compute_snr(pure_signal, noisy_signal):
    signal_to_noise_ratio = 10 * (np.log10(np.std(pure_signal)/np.std(noisy_signal-pure_signal)))
    return signal_to_noise_ratio


class Bp_Kalman_filter():

    def __init__(self):
        self.Gpu = torch.device("cuda")
        self.model = BpFilter().to(self.Gpu)
        model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\滤波模型\透射谱的滤波模型\BP_KF.pt"
        self.model.load_state_dict(torch.load(model_save_path))

    def predict(self, index, CH4_noisy_spectral, CH4_no_noise_spectral):
        data = plain_kalman(index, CH4_noisy_spectral, CH4_no_noise_spectral)
        data = array_to_tensor(data)
        data = data.reshape(1111, 1)
        prediction = self.model(data)
        prediction = prediction.cpu().detach().numpy()
        return prediction


class SG_filter():

    def __init__(self):
        self.window_length = 11
        self.polynomial = 2

    def predict(self, data):
        prediction = savgol_filter(data, self.window_length, self.polynomial)
        return prediction


class Wavelet_filter():

    def __init__(self):
        self.wavelet = pywt.Wavelet('sym5')  # 选用Daubechies8小波
        self.threshold1 = 0.9  # threshold coefficient

    def predict(self, data):
        maxlev = pywt.dwt_max_level(len(data), self.wavelet.dec_len)  # maximum level is 6
        print("maximum level is " + str(maxlev))
        # Decompose into wavelet components, to the level selected:
        coefficients = pywt.wavedec(data, self.wavelet, level=maxlev)  # 将信号进行小波分解
        for i in range(1, len(coefficients)):
            coefficients[i] = pywt.threshold(coefficients[i],
                                             self.threshold1 * max(coefficients[i]))  # 将噪声滤波
        reconstructed_signal1 = pywt.waverec(coefficients, self.wavelet)[:1111]  # 将信号进行小波重构
        return reconstructed_signal1


if __name__ == "__main__":

    """模拟数据 （透射谱）"""
    no_noise_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH_nonoise_spectral.npy"
    CH4_no_noise_spectral = np.load(no_noise_path)  # (1000, 1111) 透射谱（吸收谱）
    noisy_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH_noisy_spectral.npy"
    CH4_noisy_spectral = np.load(noisy_path)  # (1000, 1111) 透射谱（吸收谱）

    After_KF_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_afterKF.npy"
    CH4_KF = np.load(After_KF_path)

    index = 999
    Bp_kalman_filter = Bp_Kalman_filter()
    S_G_filter = SG_filter()
    Wave_filter = Wavelet_filter()

    noisy_signal = CH4_noisy_spectral[index]
    pure_signal = CH4_no_noise_spectral[index]
    kf_result = CH4_KF[index]
    bp_result = Bp_kalman_filter.predict(index, CH4_noisy_spectral, CH4_no_noise_spectral)
    sg_result = S_G_filter.predict(CH4_noisy_spectral[index])
    wt_result = Wave_filter.predict(CH4_noisy_spectral[index])

    path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\结果数据\noisy_signal.txt"
    np.savetxt(path1, noisy_signal)

    path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\结果数据\pure_signal.txt"
    np.savetxt(path2, pure_signal)

    path3 = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\结果数据\kf_result.txt"
    np.savetxt(path3, kf_result)

    path4 = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\结果数据\bp_result.txt"
    np.savetxt(path4, bp_result)

    path5 = r"D:/PYHTON\python3.7/DeepLearningProgram/深度学习滤波器/结果数据/sg_result.txt"
    np.savetxt(path5, sg_result)

    path6 = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\结果数据\wt_result.txt"
    np.savetxt(path6, wt_result)

    original_snr = compute_snr(pure_signal, noisy_signal)
    kf_snr = compute_snr(pure_signal, kf_result)
    bp_snr = compute_snr(pure_signal, bp_result)
    sg_snr = compute_snr(pure_signal, sg_result)
    wt_snr = compute_snr(pure_signal, wt_result)

    plt.figure()
    plt.plot(noisy_signal, label="without any filtering, SNR={:.4f}".format(original_snr))
    plt.plot(kf_result, alpha=1, label="Kalman filtering result, SNR={:.4f}".format(kf_snr))
    plt.plot(bp_result, alpha=1, label="BP-KF result, SNR={:.4f}".format(bp_snr))
    plt.plot(sg_result, alpha=1, label="S-G filtering result, SNR={:.4f}".format(sg_snr))
    plt.plot(wt_result, alpha=1, label="wavelet denosing result, SNR={:.4f}".format(wt_snr))
    plt.plot(pure_signal, linewidth=3, label="pure signal")
    plt.legend()
    plt.show()


