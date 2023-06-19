import numpy as np
import librosa
import os
import torch
import pesq
from pystoi import stoi

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)



def SI_SDR(target, preds):
    EPS = 1e-8
    alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    target_scaled = alpha * target
    noise = target_scaled - preds
    si_sdr_value = (np.sum(target_scaled ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 10 * np.log10(si_sdr_value)
    return si_sdr_value


def get_power(x, nfft):
    S = librosa.stft(x, nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


def LSD(x_hr, x_pr):
    S1 = get_power(x_hr, nfft=2048) # 2048
    S2 = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high

# 读取文件夹1中的音频文件
folder1 = 'E:\\Clearning\\cursor\\bone\\enhanced\\dpcrnbweopus\\yuanshi'
audio_files1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.wav')]

# 读取文件夹2中的音频文件
folder2 = 'E:\\Clearning\\cursor\\bone\\enhanced\\dpcrnbweopus\\nostream'
audio_files2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.wav')]

# 计算SI_SDR和LSD
si_sdr_list = []
lsd_list = []
lsd_high_list = []
pesq_list = []
stoi_list = []
for audio_file1, audio_file2 in zip(audio_files1, audio_files2):
    x_hr, sr = librosa.load(audio_file1, sr=None)
    x_pr, sr = librosa.load(audio_file2, sr=None)
    if len(x_hr) > len(x_pr):
        x_hr = x_hr[:len(x_pr)]
    else:
        x_pr = x_pr[:len(x_hr)]
    x_hr1 = torch.from_numpy(x_hr)
    x_pr1 = torch.from_numpy(x_pr)
    x_hr1 = torch.unsqueeze(x_hr1, 0)
    x_pr1 = torch.unsqueeze(x_pr1, 0)
    torch_si_snr = si_snr(x_pr1,x_hr1)
    si_sdr_value = SI_SDR(x_hr, x_pr)
    lsd, lsd_high = LSD(x_hr, x_pr)
    pesq_value = pesq.pesq(sr, x_hr, x_pr, 'wb')
    stoi_value = stoi(x_hr, x_pr, sr, extended=False)
    si_sdr_list.append(si_sdr_value)
    lsd_list.append(lsd)
    lsd_high_list.append(lsd_high)
    pesq_list.append(pesq_value)
    stoi_list.append(stoi_value)
    print(f"audio_file1: { audio_file1 }, SI_SDR: {si_sdr_value}, LSD: {lsd}, LSD_high: {lsd_high}, torch_si_snr: {torch_si_snr}, PESQ: {pesq_value}, STOI: {stoi_value}")

# 计算SI_SDR和LSD的平均值
si_sdr_mean = np.mean(si_sdr_list)
lsd_mean = np.mean(lsd_list)
lsd_high_mean = np.mean(lsd_high_list)
pesq_mean = np.mean(pesq_list)
stoi_mean = np.mean(stoi_list)
print(f"SI_SDR平均值: {si_sdr_mean}, LSD平均值: {lsd_mean}, LSD_high平均值: {lsd_high_mean}, PESQ平均值: {pesq_mean}, STOI平均值: {stoi_mean}")