import pickle
import numpy as np
import matplotlib.pyplot as plt


def clip(path, name, save):
    file = open(path + name, "rb")
    label = pickle.load(file)
    label = np.array(label)
    label = label[:, 1097:3097]
    file = open(save + name, "wb")
    pickle.dump(label, file)
    file.close()
    print("file saved !")


file_path = r"D:\PYHTON\python3.7\DeepLearningProgram\TraceGasSensoring\PKL格式训练数据汇总"
save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\剪裁后PKL数据"
file_name_1 = r"\CH4_0dB.pkl"
file_name_2 = r"\CH4_10dB.pkl"
file_name_3 = r"\CH4_20dB.pkl"
file_name_4 = r"\C2H2_0dB.pkl"
file_name_5 = r"\C2H2_10dB.pkl"
file_name_6 = r"\C2H2_20dB.pkl"

clip(file_path, file_name_1, save_path)
clip(file_path, file_name_2, save_path)
clip(file_path, file_name_3, save_path)
clip(file_path, file_name_4, save_path)
clip(file_path, file_name_5, save_path)
clip(file_path, file_name_6, save_path)