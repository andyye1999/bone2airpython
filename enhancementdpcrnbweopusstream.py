import argparse
import json
import os
import time
import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf
from src.generator import Generatorseanet1
from util.utils import initialize_config, load_checkpoint
from model.streamdpcrn import DPCRN_RT,init_hidden

"""
Parameters
"""
parser = argparse.ArgumentParser("seanet: BWE")
parser.add_argument("-C", "--config", default= "F:\\yhc\\bone\\config\\enhancement\\dpcrnbweopus.json", type=str, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
parser.add_argument("-O", "--output_dir", default= "F:\\yhc\\bone\\enhanced\\dpcrnbweopus2", type=str, help="Where are audio save.")
parser.add_argument("-M", "--model_checkpoint_path", default= "F:\\yhc\\bone\\dpcrnbweopus2\\checkpoints\\best_model.tar", type=str, help="Checkpoint.")
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
model = initialize_config(config["model"])
model1 = DPCRN_RT()

model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
Conv_dict = model.state_dict()
Stream_dict = model1.state_dict()
Stream_dict["encoder.conv_1.Conv2d.weight"] = Conv_dict["encoder.conv_1.weight"]
Stream_dict["encoder.conv_1.Conv2d.bias"] = Conv_dict["encoder.conv_1.bias"]
Stream_dict["encoder.bn_1.weight"] = Conv_dict["encoder.bn_1.weight"]
Stream_dict["encoder.bn_1.bias"] = Conv_dict["encoder.bn_1.bias"]
Stream_dict["encoder.bn_1.running_mean"] = Conv_dict["encoder.bn_1.running_mean"]
Stream_dict["encoder.bn_1.running_var"] = Conv_dict["encoder.bn_1.running_var"]
Stream_dict["encoder.bn_1.num_batches_tracked"] = Conv_dict["encoder.bn_1.num_batches_tracked"]
Stream_dict["encoder.act_1.weight"] = Conv_dict["encoder.act_1.weight"]
Stream_dict["encoder.conv_2.Conv2d.weight"] = Conv_dict["encoder.conv_2.weight"]
Stream_dict["encoder.conv_2.Conv2d.bias"] = Conv_dict["encoder.conv_2.bias"]
Stream_dict["encoder.bn_2.weight"] = Conv_dict["encoder.bn_2.weight"]
Stream_dict["encoder.bn_2.bias"] = Conv_dict["encoder.bn_2.bias"]
Stream_dict["encoder.bn_2.running_mean"] = Conv_dict["encoder.bn_2.running_mean"]
Stream_dict["encoder.bn_2.running_var"] = Conv_dict["encoder.bn_2.running_var"]
Stream_dict["encoder.bn_2.num_batches_tracked"] = Conv_dict["encoder.bn_2.num_batches_tracked"]
Stream_dict["encoder.act_2.weight"] = Conv_dict["encoder.act_2.weight"]
Stream_dict["encoder.conv_3.Conv2d.weight"] = Conv_dict["encoder.conv_3.weight"]
Stream_dict["encoder.conv_3.Conv2d.bias"] = Conv_dict["encoder.conv_3.bias"]
Stream_dict["encoder.bn_3.weight"] = Conv_dict["encoder.bn_3.weight"]
Stream_dict["encoder.bn_3.bias"] = Conv_dict["encoder.bn_3.bias"]
Stream_dict["encoder.bn_3.running_mean"] = Conv_dict["encoder.bn_3.running_mean"]
Stream_dict["encoder.bn_3.running_var"] = Conv_dict["encoder.bn_3.running_var"]
Stream_dict["encoder.bn_3.num_batches_tracked"] = Conv_dict["encoder.bn_3.num_batches_tracked"]
Stream_dict["encoder.act_3.weight"] = Conv_dict["encoder.act_3.weight"]
Stream_dict["encoder.conv_4.Conv2d.weight"] = Conv_dict["encoder.conv_4.weight"]
Stream_dict["encoder.conv_4.Conv2d.bias"] = Conv_dict["encoder.conv_4.bias"]
Stream_dict["encoder.bn_4.weight"] = Conv_dict["encoder.bn_4.weight"]
Stream_dict["encoder.bn_4.bias"] = Conv_dict["encoder.bn_4.bias"]
Stream_dict["encoder.bn_4.running_mean"] = Conv_dict["encoder.bn_4.running_mean"]
Stream_dict["encoder.bn_4.running_var"] = Conv_dict["encoder.bn_4.running_var"]
Stream_dict["encoder.bn_4.num_batches_tracked"] = Conv_dict["encoder.bn_4.num_batches_tracked"]
Stream_dict["encoder.act_4.weight"] = Conv_dict["encoder.act_4.weight"]
Stream_dict["encoder.conv_5.Conv2d.weight"] = Conv_dict["encoder.conv_5.weight"]
Stream_dict["encoder.conv_5.Conv2d.bias"] = Conv_dict["encoder.conv_5.bias"]
Stream_dict["encoder.bn_5.weight"] = Conv_dict["encoder.bn_5.weight"]
Stream_dict["encoder.bn_5.bias"] = Conv_dict["encoder.bn_5.bias"]
Stream_dict["encoder.bn_5.running_mean"] = Conv_dict["encoder.bn_5.running_mean"]
Stream_dict["encoder.bn_5.running_var"] = Conv_dict["encoder.bn_5.running_var"]
Stream_dict["encoder.bn_5.num_batches_tracked"] = Conv_dict["encoder.bn_5.num_batches_tracked"]
Stream_dict["encoder.act_5.weight"] = Conv_dict["encoder.act_5.weight"]

Stream_dict["dprnn_1.intra_rnn.weight_ih_l0"] = Conv_dict["dprnn_1.intra_rnn.weight_ih_l0"]
Stream_dict["dprnn_1.intra_rnn.weight_hh_l0"] = Conv_dict["dprnn_1.intra_rnn.weight_hh_l0"]
Stream_dict["dprnn_1.intra_rnn.bias_ih_l0"] = Conv_dict["dprnn_1.intra_rnn.bias_ih_l0"]
Stream_dict["dprnn_1.intra_rnn.bias_hh_l0"] = Conv_dict["dprnn_1.intra_rnn.bias_hh_l0"]
Stream_dict["dprnn_1.intra_rnn.weight_ih_l0_reverse"] = Conv_dict["dprnn_1.intra_rnn.weight_ih_l0_reverse"]
Stream_dict["dprnn_1.intra_rnn.weight_hh_l0_reverse"] = Conv_dict["dprnn_1.intra_rnn.weight_hh_l0_reverse"]
Stream_dict["dprnn_1.intra_rnn.bias_ih_l0_reverse"] = Conv_dict["dprnn_1.intra_rnn.bias_ih_l0_reverse"]
Stream_dict["dprnn_1.intra_rnn.bias_hh_l0_reverse"] = Conv_dict["dprnn_1.intra_rnn.bias_hh_l0_reverse"]
Stream_dict["dprnn_1.intra_fc.weight"] = Conv_dict["dprnn_1.intra_fc.weight"]
Stream_dict["dprnn_1.intra_fc.bias"] = Conv_dict["dprnn_1.intra_fc.bias"]
Stream_dict["dprnn_1.intra_ln.weight"] = Conv_dict["dprnn_1.intra_ln.weight"]
Stream_dict["dprnn_1.intra_ln.bias"] = Conv_dict["dprnn_1.intra_ln.bias"]

Stream_dict["dprnn_1.intra_ln.running_mean"] = Conv_dict["dprnn_1.intra_ln.running_mean"]
Stream_dict["dprnn_1.intra_ln.running_var"] = Conv_dict["dprnn_1.intra_ln.running_var"]
Stream_dict["dprnn_1.intra_ln.num_batches_tracked"] = Conv_dict["dprnn_1.intra_ln.num_batches_tracked"]
Stream_dict["dprnn_1.inter_rnn.weight_ih_l0"] = Conv_dict["dprnn_1.inter_rnn.weight_ih_l0"]

Stream_dict["dprnn_1.inter_rnn.weight_hh_l0"] = Conv_dict["dprnn_1.inter_rnn.weight_hh_l0"]
Stream_dict["dprnn_1.inter_rnn.bias_ih_l0"] = Conv_dict["dprnn_1.inter_rnn.bias_ih_l0"]
Stream_dict["dprnn_1.inter_rnn.bias_hh_l0"] = Conv_dict["dprnn_1.inter_rnn.bias_hh_l0"]
Stream_dict["dprnn_1.inter_fc.weight"] = Conv_dict["dprnn_1.inter_fc.weight"]
Stream_dict["dprnn_1.inter_fc.bias"] = Conv_dict["dprnn_1.inter_fc.bias"]
Stream_dict["dprnn_1.inter_ln.weight"] = Conv_dict["dprnn_1.inter_ln.weight"]
Stream_dict["dprnn_1.inter_ln.bias"] = Conv_dict["dprnn_1.inter_ln.bias"]

Stream_dict["dprnn_1.inter_ln.running_mean"] = Conv_dict["dprnn_1.inter_ln.running_mean"]
Stream_dict["dprnn_1.inter_ln.running_var"] = Conv_dict["dprnn_1.inter_ln.running_var"]
Stream_dict["dprnn_1.inter_ln.num_batches_tracked"] = Conv_dict["dprnn_1.inter_ln.num_batches_tracked"]

Stream_dict["real_decoder.real_dconv_1.Conv2d.weight"] = torch.flip(Conv_dict["real_decoder.real_dconv_1.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["real_decoder.real_dconv_1.Conv2d.bias"] = Conv_dict["real_decoder.real_dconv_1.bias"]
Stream_dict["real_decoder.real_bn_1.weight"] = Conv_dict["real_decoder.real_bn_1.weight"]
Stream_dict["real_decoder.real_bn_1.bias"] = Conv_dict["real_decoder.real_bn_1.bias"]
Stream_dict["real_decoder.real_bn_1.running_mean"] = Conv_dict["real_decoder.real_bn_1.running_mean"]
Stream_dict["real_decoder.real_bn_1.running_var"] = Conv_dict["real_decoder.real_bn_1.running_var"]
Stream_dict["real_decoder.real_bn_1.num_batches_tracked"] = Conv_dict["real_decoder.real_bn_1.num_batches_tracked"]
Stream_dict["real_decoder.real_act_1.weight"] = Conv_dict["real_decoder.real_act_1.weight"]
Stream_dict["real_decoder.real_dconv_2.Conv2d.weight"] = torch.flip(Conv_dict["real_decoder.real_dconv_2.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["real_decoder.real_dconv_2.Conv2d.bias"] = Conv_dict["real_decoder.real_dconv_2.bias"]
Stream_dict["real_decoder.real_bn_2.weight"] = Conv_dict["real_decoder.real_bn_2.weight"]
Stream_dict["real_decoder.real_bn_2.bias"] = Conv_dict["real_decoder.real_bn_2.bias"]
Stream_dict["real_decoder.real_bn_2.running_mean"] = Conv_dict["real_decoder.real_bn_2.running_mean"]
Stream_dict["real_decoder.real_bn_2.running_var"] = Conv_dict["real_decoder.real_bn_2.running_var"]
Stream_dict["real_decoder.real_bn_2.num_batches_tracked"] = Conv_dict["real_decoder.real_bn_2.num_batches_tracked"]
Stream_dict["real_decoder.real_act_2.weight"] = Conv_dict["real_decoder.real_act_2.weight"]
Stream_dict["real_decoder.real_dconv_3.Conv2d.weight"] = torch.flip(
    Conv_dict["real_decoder.real_dconv_3.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["real_decoder.real_dconv_3.Conv2d.bias"] = Conv_dict["real_decoder.real_dconv_3.bias"]
Stream_dict["real_decoder.real_bn_3.weight"] = Conv_dict["real_decoder.real_bn_3.weight"]
Stream_dict["real_decoder.real_bn_3.bias"] = Conv_dict["real_decoder.real_bn_3.bias"]
Stream_dict["real_decoder.real_bn_3.running_mean"] = Conv_dict["real_decoder.real_bn_3.running_mean"]
Stream_dict["real_decoder.real_bn_3.running_var"] = Conv_dict["real_decoder.real_bn_3.running_var"]
Stream_dict["real_decoder.real_bn_3.num_batches_tracked"] = Conv_dict["real_decoder.real_bn_3.num_batches_tracked"]
Stream_dict["real_decoder.real_act_3.weight"] = Conv_dict["real_decoder.real_act_3.weight"]
Stream_dict["real_decoder.real_dconv_4.Conv2d.weight"] = torch.flip(
    Conv_dict["real_decoder.real_dconv_4.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["real_decoder.real_dconv_4.Conv2d.bias"] = Conv_dict["real_decoder.real_dconv_4.bias"]
Stream_dict["real_decoder.real_bn_4.weight"] = Conv_dict["real_decoder.real_bn_4.weight"]
Stream_dict["real_decoder.real_bn_4.bias"] = Conv_dict["real_decoder.real_bn_4.bias"]
Stream_dict["real_decoder.real_bn_4.running_mean"] = Conv_dict["real_decoder.real_bn_4.running_mean"]
Stream_dict["real_decoder.real_bn_4.running_var"] = Conv_dict["real_decoder.real_bn_4.running_var"]
Stream_dict["real_decoder.real_bn_4.num_batches_tracked"] = Conv_dict["real_decoder.real_bn_4.num_batches_tracked"]
Stream_dict["real_decoder.real_act_4.weight"] = Conv_dict["real_decoder.real_act_4.weight"]
Stream_dict["real_decoder.real_dconv_5.Conv2d.weight"] = torch.flip(
    Conv_dict["real_decoder.real_dconv_5.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["real_decoder.real_dconv_5.Conv2d.bias"] = Conv_dict["real_decoder.real_dconv_5.bias"]
Stream_dict["real_decoder.real_bn_5.weight"] = Conv_dict["real_decoder.real_bn_5.weight"]
Stream_dict["real_decoder.real_bn_5.bias"] = Conv_dict["real_decoder.real_bn_5.bias"]
Stream_dict["real_decoder.real_bn_5.running_mean"] = Conv_dict["real_decoder.real_bn_5.running_mean"]
Stream_dict["real_decoder.real_bn_5.running_var"] = Conv_dict["real_decoder.real_bn_5.running_var"]
Stream_dict["real_decoder.real_bn_5.num_batches_tracked"] = Conv_dict["real_decoder.real_bn_5.num_batches_tracked"]
Stream_dict["real_decoder.real_act_5.weight"] = Conv_dict["real_decoder.real_act_5.weight"]

Stream_dict["imag_decoder.imag_dconv_1.Conv2d.weight"] = torch.flip(
    Conv_dict["imag_decoder.imag_dconv_1.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["imag_decoder.imag_dconv_1.Conv2d.bias"] = Conv_dict["imag_decoder.imag_dconv_1.bias"]
Stream_dict["imag_decoder.imag_bn_1.weight"] = Conv_dict["imag_decoder.imag_bn_1.weight"]
Stream_dict["imag_decoder.imag_bn_1.bias"] = Conv_dict["imag_decoder.imag_bn_1.bias"]
Stream_dict["imag_decoder.imag_bn_1.running_mean"] = Conv_dict["imag_decoder.imag_bn_1.running_mean"]
Stream_dict["imag_decoder.imag_bn_1.running_var"] = Conv_dict["imag_decoder.imag_bn_1.running_var"]
Stream_dict["imag_decoder.imag_bn_1.num_batches_tracked"] = Conv_dict["imag_decoder.imag_bn_1.num_batches_tracked"]
Stream_dict["imag_decoder.imag_act_1.weight"] = Conv_dict["imag_decoder.imag_act_1.weight"]
Stream_dict["imag_decoder.imag_dconv_2.Conv2d.weight"] = torch.flip(
    Conv_dict["imag_decoder.imag_dconv_2.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["imag_decoder.imag_dconv_2.Conv2d.bias"] = Conv_dict["imag_decoder.imag_dconv_2.bias"]
Stream_dict["imag_decoder.imag_bn_2.weight"] = Conv_dict["imag_decoder.imag_bn_2.weight"]
Stream_dict["imag_decoder.imag_bn_2.bias"] = Conv_dict["imag_decoder.imag_bn_2.bias"]
Stream_dict["imag_decoder.imag_bn_2.running_mean"] = Conv_dict["imag_decoder.imag_bn_2.running_mean"]
Stream_dict["imag_decoder.imag_bn_2.running_var"] = Conv_dict["imag_decoder.imag_bn_2.running_var"]
Stream_dict["imag_decoder.imag_bn_2.num_batches_tracked"] = Conv_dict["imag_decoder.imag_bn_2.num_batches_tracked"]
Stream_dict["imag_decoder.imag_act_2.weight"] = Conv_dict["imag_decoder.imag_act_2.weight"]
Stream_dict["imag_decoder.imag_dconv_3.Conv2d.weight"] = torch.flip(
    Conv_dict["imag_decoder.imag_dconv_3.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["imag_decoder.imag_dconv_3.Conv2d.bias"] = Conv_dict["imag_decoder.imag_dconv_3.bias"]
Stream_dict["imag_decoder.imag_bn_3.weight"] = Conv_dict["imag_decoder.imag_bn_3.weight"]
Stream_dict["imag_decoder.imag_bn_3.bias"] = Conv_dict["imag_decoder.imag_bn_3.bias"]
Stream_dict["imag_decoder.imag_bn_3.running_mean"] = Conv_dict["imag_decoder.imag_bn_3.running_mean"]
Stream_dict["imag_decoder.imag_bn_3.running_var"] = Conv_dict["imag_decoder.imag_bn_3.running_var"]
Stream_dict["imag_decoder.imag_bn_3.num_batches_tracked"] = Conv_dict["imag_decoder.imag_bn_3.num_batches_tracked"]
Stream_dict["imag_decoder.imag_act_3.weight"] = Conv_dict["imag_decoder.imag_act_3.weight"]
Stream_dict["imag_decoder.imag_dconv_4.Conv2d.weight"] = torch.flip(
    Conv_dict["imag_decoder.imag_dconv_4.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["imag_decoder.imag_dconv_4.Conv2d.bias"] = Conv_dict["imag_decoder.imag_dconv_4.bias"]
Stream_dict["imag_decoder.imag_bn_4.weight"] = Conv_dict["imag_decoder.imag_bn_4.weight"]
Stream_dict["imag_decoder.imag_bn_4.bias"] = Conv_dict["imag_decoder.imag_bn_4.bias"]
Stream_dict["imag_decoder.imag_bn_4.running_mean"] = Conv_dict["imag_decoder.imag_bn_4.running_mean"]
Stream_dict["imag_decoder.imag_bn_4.running_var"] = Conv_dict["imag_decoder.imag_bn_4.running_var"]
Stream_dict["imag_decoder.imag_bn_4.num_batches_tracked"] = Conv_dict["imag_decoder.imag_bn_4.num_batches_tracked"]
Stream_dict["imag_decoder.imag_act_4.weight"] = Conv_dict["imag_decoder.imag_act_4.weight"]
Stream_dict["imag_decoder.imag_dconv_5.Conv2d.weight"] = torch.flip(
    Conv_dict["imag_decoder.imag_dconv_5.weight"].permute([1, 0, 2, 3]), dims=[-2, -1])
Stream_dict["imag_decoder.imag_dconv_5.Conv2d.bias"] = Conv_dict["imag_decoder.imag_dconv_5.bias"]
Stream_dict["imag_decoder.imag_bn_5.weight"] = Conv_dict["imag_decoder.imag_bn_5.weight"]
Stream_dict["imag_decoder.imag_bn_5.bias"] = Conv_dict["imag_decoder.imag_bn_5.bias"]
Stream_dict["imag_decoder.imag_bn_5.running_mean"] = Conv_dict["imag_decoder.imag_bn_5.running_mean"]
Stream_dict["imag_decoder.imag_bn_5.running_var"] = Conv_dict["imag_decoder.imag_bn_5.running_var"]
Stream_dict["imag_decoder.imag_bn_5.num_batches_tracked"] = Conv_dict["imag_decoder.imag_bn_5.num_batches_tracked"]
Stream_dict["imag_decoder.imag_act_5.weight"] = Conv_dict["imag_decoder.imag_act_5.weight"]

#
model1.load_state_dict(Stream_dict)
model.to(device)
model.eval()
model1.to(device)
model1.eval()
cache = torch.zeros([1, 2, 1, 201])
cache1 = torch.zeros([1, 32, 1, 100])
cache2 = torch.zeros([1, 32, 1, 50])
cache3 = torch.zeros([1, 32, 1, 50])
cache4 = torch.zeros([1, 64, 1, 50])
cache5 = torch.zeros([1, 256, 1, 50])
cache6 = torch.zeros([1, 128, 1, 50])
cache7 = torch.zeros([1, 64, 1, 50])
cache8 = torch.zeros([1, 64, 1, 50])
cache9 = torch.zeros([1, 64, 1, 100])
cache10 = torch.zeros([1, 256, 1, 50])
cache11 = torch.zeros([1, 128, 1, 50])
cache12 = torch.zeros([1, 64, 1, 50])
cache13 = torch.zeros([1, 64, 1, 50])
cache14 = torch.zeros([1, 64, 1, 100])
intra_init_hidden = init_hidden(2, 1, 64)
inter_init_hidden = init_hidden(1, 50, 128)
"""
Enhancement
"""
sample_length = config["custom"]["sample_length"]
for mixture,  name in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0
    mixture = mixture.to(device)  # [1, 1, T]
    chunk = torch.squeeze(mixture, dim=1)
    input_stft = torch.stft(chunk, n_fft=400, hop_length=160, win_length=320)  # (Bs, F, T, 2)
    Bs, F, T, C = input_stft.shape
    output_stft = torch.empty((Bs, F, 0, C), dtype=input_stft.dtype, device=input_stft.device)
    for i in range(T):
        time_start = time.time()  # 记录开始时间
        output1, cache, cache1, cache2, cache3, cache4, intra_init_hidden, inter_init_hidden, cache5, cache6, cache7, cache8, cache9, cache10, cache11, cache12, cache13, cache14 = model1(
            input_stft[:, :, i:i + 1], cache, cache1, cache2, cache3, cache4, intra_init_hidden, inter_init_hidden, cache5,
            cache6, cache7, cache8, cache9, cache10, cache11, cache12, cache13, cache14)
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum)
        output_stft = torch.cat((output_stft, output1), dim=2)


    out_wav = torch.istft(output_stft, n_fft=400,
                          hop_length=160,
                          win_length=320)
    output = torch.unsqueeze(out_wav, 1)
    enhanced = output
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

    # enhanced_chunks = []
    # for chunk in mixture_chunks:
    #     chunk = torch.squeeze(chunk, dim=1)
    #     input_stft = torch.stft(chunk, n_fft=400, hop_length=160, win_length=320)  # (Bs, F, T, 2)
    #
    #     output_stft = model(input_stft)
    #     out_wav = torch.istft(output_stft, n_fft=400,
    #                           hop_length=160,
    #                           win_length=320)
    #     output = torch.unsqueeze(out_wav, 1)
    #     # enhanced_speech = enhanced_speech * max_bone
    #     enhanced_chunks.append(output)
    # enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
    # enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

    # clean = clean * max_air
    enhanced = enhanced.detach().cpu()
    enhanced = enhanced.reshape(-1).numpy()
    # clean = clean.cpu().numpy().reshape(-1)
    # mixture = mixture.cpu().numpy().reshape(-1)
    output_path = os.path.join(output_dir, f"{name}.wav")
    # librosa.output.write_wav(output_path, enhanced, sr=16000)
    sf.write(output_path, enhanced, 16000)
