# 导入需要的库
import librosa
import numpy as np

# 定义计算LSD的函数
def lsd(audio1, audio2):
    # 计算音频1的短时傅里叶变换
    stft1 = librosa.stft(audio1)
    # 计算音频2的短时傅里叶变换
    stft2 = librosa.stft(audio2)
    # 计算音频1的功率谱
    power1 = np.abs(stft1)**2
    # 计算音频2的功率谱
    power2 = np.abs(stft2)**2
    # 计算两个音频的LSD
    lsd = np.sqrt(np.sum((np.log10(power1) - np.log10(power2))**2))
    return lsd

# 定义计算LSD-HF的函数
def lsd_hf(audio1, audio2):
    # 定义高频区间
    hf_band = [2000, 8000]
    # 计算音频1的短时傅里叶变换
    stft1 = librosa.stft(audio1)
    # 计算音频2的短时傅里叶变换
    stft2 = librosa.stft(audio2)
    # 计算音频1的功率谱
    power1 = np.abs(stft1)**2
    # 计算音频2的功率谱
    power2 = np.abs(stft2)**2
    # 找到高频区间的索引
    hf_idx = np.where((librosa.fft_frequencies() >= hf_band[0]) & (librosa.fft_frequencies() <= hf_band[1]))[0]
    # 计算两个音频在高频区间的LSD
    lsd_hf = np.sqrt(np.sum((np.log10(power1[hf_idx, :]) - np.log10(power2[hf_idx, :]))**2))
    return lsd_hf


input_folder1 = 'E:\\Clearning\\cursor\\bone\\enhanced\\pesq\\eben7\\air_clk82.wav'
output_folder = 'E:\\Clearning\\cursor\\bone\\enhanced\\pesq\\eben7\\bone_clk82_eben7.wav'
x,_ = librosa.load(input_folder1)
y,_ = librosa.load(output_folder)

lsd1 = lsd(x,y)
print(lsd1)

