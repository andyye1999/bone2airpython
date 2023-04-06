import argparse
import json
import os

import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.utils import initialize_config, load_checkpoint

"""
Parameters
"""
parser = argparse.ArgumentParser("Wave-U-Net: Speech Enhancement")
parser.add_argument("-C", "--config", default= "/home/dsp/yhc/bone/config/enhancement/loglstmenhance.json", type=str, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
parser.add_argument("-O", "--output_dir", default= "/home/dsp/yhc/bone/enhanced", type=str, help="Where are audio save.")
parser.add_argument("-M", "--model_checkpoint_path", default= "/home/dsp/yhc/bone/loglstm/checkpoints/latest_model.tar", type=str, help="Checkpoint.")
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
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)

"""
Model
"""
model = initialize_config(config["model"])
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

"""
Enhancement
"""
sample_length = config["custom"]["sample_length"]
for mixture, name in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0

    mixture = mixture.to(device)  # [1, 1, T]

    # The input of the model should be fixed length.
    if mixture.size(-1) % sample_length != 0:
        padded_length = sample_length - (mixture.size(-1) % sample_length)
        mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)

    assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
    mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

    enhanced_chunks = []
    for chunk in mixture_chunks:
        mu_bone = -9.981625  # bone   # air -3.48766 9.9217825
        sigma_bone = 15.002887
        mu_air = -6.493551
        sigma_air = 15.735691
        # calculate maximum and minimum values of X for each batch
        # max_val_bone, _ = torch.max(torch.abs(mixture), dim=2, keepdim=True)
        max_val_bone = torch.max(torch.abs(chunk))

        bone_normalized = chunk / max_val_bone
        # bone_normalized = mixture
        bone_normalized = torch.squeeze(bone_normalized, dim=1)
        pred_stft = torch.stft(bone_normalized, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        pred_mag = (pred_mag - mu_bone) / sigma_bone
        input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
        enhance = model(input).detach().cpu()
        # enhance = model(input)

        mag = enhance.permute(0, 2, 1)
        mag = mag * sigma_air + mu_air
        mag = torch.exp(mag / 2)  # 将对数幅度谱转换成幅度谱
        pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        output = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320,
                            hop_length=160,
                            win_length=320)
        # if torch.isnan(output).any():
        #     print("nan error")
        #     input("press enter ")
        output = torch.unsqueeze(output, 1)
        output = output * max_val_bone
        # enhanced_chunks.append(self.model(chunk).detach().cpu())
        enhanced_chunks.append(output)

    enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
    enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

    # enhanced = enhanced.reshape(-1).numpy()
    enhanced = enhanced.reshape(-1).detach().numpy()
    output_path = os.path.join(output_dir, f"{name}.wav")
    librosa.output.write_wav(output_path, enhanced, sr=16000)
