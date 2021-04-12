# author: SaKuRa Pop
# data: 2021/4/9 11:10
import pickle
import xlrd
import numpy as np

CH4_simulated_no_noise = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_experiment_nonoise.xlsx"
CH4_simulated_no_noise_npy = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_sim_no_noise.npy"

CH4_simulated_noisy = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_experiment_withnoise.xlsx"
CH4_simulated_noisy_npy = r"D:\PYHTON\python3.7\DeepLearningProgram\深度学习滤波器\1000组模拟数据\提供给模型的数据\模拟数据\CH4_sim_withnoise.npy"


def xlsx_to_pkl(open_file_path, save_file_path):
    data = xlrd.open_workbook(open_file_path)
    table = data.sheet_by_name('Sheet1')
    table_array = np.zeros((1000, 1111))
    for i in range(1000):
        for j in range(1111):
            table_array[i, j] = table.cell_value(i, j)

    data_x = table_array
    file_x = np.save(save_file_path, data_x)
    print("saved!")


xlsx_to_pkl(CH4_simulated_no_noise, CH4_simulated_no_noise_npy)
xlsx_to_pkl(CH4_simulated_noisy, CH4_simulated_noisy_npy)