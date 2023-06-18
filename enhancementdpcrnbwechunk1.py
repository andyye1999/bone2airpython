import argparse
import json
import os
import time
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf
from src.generator import Generatorseanet1
from util.utils import initialize_config, load_checkpoint

"""
Parameters
"""
parser = argparse.ArgumentParser("dpcrn: BWE")
parser.add_argument("-C", "--config", default= "F:\\yhc\\bone\\config\\enhancement\\dpcrnbweopus.json", type=str, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
parser.add_argument("-O", "--output_dir", default= "F:\\yhc\\bone\\enhanced\\dpcrnbweopus", type=str, help="Where are audio save.")
parser.add_argument("-M", "--model_checkpoint_path", default= "F:\\yhc\\bone\\dpcrnbweopus\\checkpoints\\best_model.tar", type=str, help="Checkpoint.")
# parser.add_argument("-M", "--model_checkpoint_path", default= "F:\\yhc\\bone\\dpcrnbwe2\\checkpoints\\model_0465.pth", type=str, help="Checkpoint.")

args = parser.parse_args()


"""
Preparation
"""
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = json.load(open(args.config))
model_checkpoint_path = args.model_checkpoint_path
output_dir = args.output_dir
assert os.path.exists(output_dir), "Enhanced directory should be exist."

"""
DataLoader
"""
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)

"""
Model
"""
model = initialize_config(config["model"])
# model = GeneratorEBEN(bands_nbr=4, pqmf_ks=32)
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

"""
Enhancement
"""
sample_length = config["custom"]["sample_length"]
frame_lenth = config["custom"]["frame_lenth"]
hop_lenth = config["custom"]["hop_lenth"]
for mixture,  name in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0
    mixture = mixture.cpu().numpy()
    mixture = np.squeeze(mixture)
    mixture = np.pad(mixture, (frame_lenth - hop_lenth, 0), 'constant', constant_values=0)
    frames = librosa.util.frame(mixture,frame_length=frame_lenth,hop_length=hop_lenth, axis=0)
    fft_window = librosa.filters.get_window('hann',frame_lenth)
    fft_window = fft_window.astype(np.float32)
    # frames = frames * fft_window
    n_frames, _ = frames.shape
    frames = torch.tensor(frames)
    frames = frames.to(device)  # [1, 1, T]
    pre_frames = []
    for frame in frames:
        frame = frame.unsqueeze(0)
        time_start = time.time()  # 记录开始时间
        input_stft = torch.stft(frame, n_fft=400, hop_length=160, win_length=320)  # (Bs, F, T, 2)

        output_stft = model(input_stft)
        out_wav = torch.istft(output_stft, n_fft=400,
                              hop_length=160,
                              win_length=320)
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum)
        output = torch.unsqueeze(out_wav, 1)
        pre_frame = output
        pre_frame = pre_frame.detach().numpy().reshape(-1)
        pre_frames.append(pre_frame[frame_lenth - hop_lenth:frame_lenth])

    # clean = clean.to(device)
    # max_bone = max_bone.to(device)
    # max_air = max_air.to(device)
    # mixture = model.cut_tensor(mixture)
    # clean = model.cut_tensor(clean)
    # if mixture.size(-1) % sample_length != 0:
    #     padded_length = sample_length - (mixture.size(-1) % sample_length)
    #     mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)
    #
    # assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
    # mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))
    #
    # enhanced_chunks = []
    # for chunk in mixture_chunks:
    #     enhanced_speech = model(chunk)
    #     # enhanced_speech = enhanced_speech * max_bone
    #     enhanced_chunks.append(enhanced_speech)
    # enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
    # enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
    enhanced = np.zeros_like(mixture)
    for i in range(n_frames):
        start = i * hop_lenth
        end = start + hop_lenth
        enhanced[start:end] = pre_frames[i]
    # enhanced = librosa.util.fix_length(np.concatenate(pre_frames),len(mixture))
    # clean = clean * max_air
    # enhanced = enhanced.detach().cpu()
    # enhanced = enhanced.reshape(-1).numpy()
    # clean = clean.cpu().numpy().reshape(-1)
    # mixture = mixture.cpu().numpy().reshape(-1)
    output_path = os.path.join(output_dir, f"{name}.wav")
    # librosa.output.write_wav(output_path, enhanced, sr=16000)
    sf.write(output_path, enhanced, 16000)
