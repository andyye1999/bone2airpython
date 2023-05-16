import argparse
import json
import os

import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf

from util.utils import initialize_config, load_checkpoint
from model.soundstream import SoundStream
"""
Parameters
"""
parser = argparse.ArgumentParser("Wave-U-Net: Speech Enhancement")
parser.add_argument("-C", "--config", default= "F:\\yhc\\bone\\config\\enhancement\\soundstream.json", type=str, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
parser.add_argument("-O", "--output_dir", default= "F:\\yhc\\bone\\enhanced\\soundstream3", type=str, help="Where are audio save.")
parser.add_argument("-M", "--model_checkpoint_path", default= "F:\\yhc\\bone\\soundstream3\\checkpoints\\best_model.tar", type=str, help="Checkpoint.")
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
# device = torch.device("cpu")
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)

"""
Model
"""
model = SoundStream(32, 32, 6, 1024, [4, 4, 5, 10])
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

"""
Enhancement
"""
sample_length = config["custom"]["sample_length"]
for mixture,  name, max_bone in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0

    mixture = mixture.to(device)  # [1, 1, T]
    # mixture = mixture.to(self.device)  # [1, 1, T]
    # clean = clean.to(device)  # [1, 1, T]
    max_bone = max_bone.to(device)
    # max_air = max_air.to(device)
    # The input of the model should be fixed length.
    if mixture.size(-1) % sample_length != 0:
        padded_length = sample_length - (mixture.size(-1) % sample_length)
        mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)

    assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
    mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

    enhanced_chunks = []
    for chunk in mixture_chunks:

        out_wav = model(chunk)
        out_wav = out_wav * max_bone
        enhanced_chunks.append(out_wav)


    enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
    enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

    # enhanced = enhanced.reshape(-1).numpy()
    enhanced = enhanced.detach().cpu()
    enhanced = enhanced.reshape(-1).numpy()
    # enhanced = enhanced.reshape(-1).detach().numpy()
    output_path = os.path.join(output_dir, f"{name}.wav")
    # librosa.output.write_wav(output_path, enhanced, sr=16000)
    sf.write(output_path, enhanced, 8000)
