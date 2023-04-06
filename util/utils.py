import importlib
import time
import os
import random
import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
import librosa


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])



def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")

def sample_pad(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    # data_a, data_b = normalize_data(data_a,data_b)
    frames_total = len(data_a)

    # 计算需要填充的帧数
    remainder = frames_total % sample_length
    if remainder != 0:
        num_frames_to_pad = sample_length - remainder
        # 使用零填充数据
        data_a = np.pad(data_a, (0, num_frames_to_pad))
        data_b = np.pad(data_b, (0, num_frames_to_pad))
        frames_total += num_frames_to_pad

    # start = np.random.randint(frames_total - sample_length + 1)
    # # print(f"Random crop from: {start}")
    # end = start + sample_length

    return data_a, data_b

def normalize_data(data_a, data_b):
    """normalize data_a and data_b by dividing by their maximum values
    """
    max_a = np.max(data_a)
    max_b = np.max(data_b)
    tmpa = 1 / max_a if max_a != 0 else 0
    tmpb = 1 / max_b if max_b != 0 else 0

    return data_a * tmpa, data_b * tmpb, max_a, max_b

def normalize_data_enhance(data_a):
    """normalize data_a and data_b by dividing by their maximum values
    """
    max_a = np.max(data_a)


    return data_a / max_a, max_a

def sample_fixed(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)
    frames_to_keep = frames_total // sample_length * sample_length

    start = np.random.randint(frames_total - frames_to_keep + 1)
    # print(f"Random crop from: {start}")
    end = start + frames_to_keep

    return data_a[start:end], data_b[start:end]

def copymag(input):
    input_stft = librosa.stft(input, n_fft=512)
    input_stft[129:257, :] = input_stft[1:129, :]
    output = librosa.istft(input_stft, length=len(input))
    return output
def copymag4(input):
    input_stft = librosa.stft(input, n_fft=512)
    input_stft[1:2,:] = 0
    input_stft[65:129, :] = input_stft[1:65, :] / 4
    input_stft[129:193, :] = input_stft[1:65, :] / 16
    input_stft[193:257, :] = input_stft[1:65, :] / 16
    output = librosa.istft(input_stft, length=len(input))
    return output

def frame(a, w, s, copy=True):
    if len(a) < w:
        return np.expand_dims(np.hstack((a, np.zeros(w - len(a)))), 0)

    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::s]

    if copy:
        return view.copy()
    else:
        return view


def pad(sig, length):
    if len(sig) < length:
        pad = length - len(sig)
        sig = np.hstack((sig, np.zeros(pad) + 0.1))
    else:
        start = random.randint(0, len(sig) - length)
        sig = sig[start:start + length]
    return sig

def overlap_add(x, win_len, hop_size, target_shape):
    # target.shape = (B, C, seq_len)
    # x.shape = (B*n_chunks, C, win_len) , n_chunks = (seq_len - hop_size)/(win_len - hop_size)
    bs, channels, seq_len = target_shape
    hann_windows = torch.ones(x.shape, device=x.device) * torch.hann_window(win_len, device=x.device)
    hann_windows[0, :, :hop_size] = 1
    hann_windows[-1, :, -hop_size:] = 1
    x *= hann_windows
    x = x.permute(1, 0, 2).reshape(bs * channels, -1, win_len).permute(0, 2, 1)  # B*C, win_len, n_chunks
    fold = torch.nn.Fold(output_size=(1, seq_len), kernel_size=(1, win_len), stride=(1, hop_size))
    x = fold(x)  # B*C, 1, 1, seq_len
    x = x.reshape(channels, bs, seq_len).permute(1, 0, 2)  # B, C, seq_len
    return x

def collate_fn(batch):
    data = torch.cat([item[0] for item in batch], dim=0).float()
    target = torch.cat([item[1] for item in batch], dim=0).float()
    return [data, target]
