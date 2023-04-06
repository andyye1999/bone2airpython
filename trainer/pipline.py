import torch
import torch.nn.functional as F


class NetFeeder(object):
    def __init__(self):
        super(NetFeeder, self).__init__()


    def __call__(self, mixture, clean):
        # max_val_bone, _ = torch.max(torch.abs(mixture), dim=2, keepdim=True)
        bone_normalized = mixture
        bone_normalized = torch.squeeze(bone_normalized, dim=1)
        pred_stft = torch.stft(bone_normalized, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        # pred_stft_real[:, 38:48, :] = pred_stft_real[:, 101:111, :]
        # pred_stft_real[:, 38:48, :] = pred_stft_real[:, 101:111, :]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        # pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # pred_mag[:, 36:71, :] = pred_mag[:, 1:36, :]
        # pred_mag[:, 71:106, :] = pred_mag[:, 1:36, :]
        # pred_mag[:, 106:141, :] = pred_mag[:, 1:36, :]
        # input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)

        # max_val_air, _ = torch.max(torch.abs(clean), dim=2, keepdim=True)
        air_normalized = clean
        air_normalized = torch.squeeze(air_normalized, dim=1)
        air_stft = torch.stft(air_normalized, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
        air_stft_real, air_stft_imag = air_stft[:, :, :, 0], air_stft[:, :, :, 1]
        air_mag = torch.sqrt(air_stft_real ** 2 + air_stft_imag ** 2 + 1e-6)
        pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
        return pred_mag, air_mag, pha

class Resynthesizer(object):
    def __init__(self):
        super(Resynthesizer, self).__init__()


    def __call__(self, enhance_mag, pha):
        real = enhance_mag * torch.cos(pha)
        imag = enhance_mag * torch.sin(pha)
        model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320,
                            hop_length=160,
                            win_length=320)
        if torch.isnan(model).any():
            print("nan error")
            input("press enter ")

        return model

def line(signal):

    # bone_normalized = torch.squeeze(signal, dim=1)
    pred_stft = torch.stft(signal, n_fft=512, hop_length=100, win_length=400)  # (Bs, F, T, 2)
    pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
    pred_stft_real[:, 1:129, :] = pred_stft_real[:, 129:257, :]
    pred_stft_imag[:, 1:129, :] = pred_stft_imag[:, 129:257, :]
    output = torch.istft(torch.cat([pred_stft_real.unsqueeze(-1), pred_stft_imag.unsqueeze(-1)], dim=-1), n_fft=512,
                        hop_length=100,
                        win_length=400)
    max_val_bone, _ = torch.max(torch.abs(output), dim=1, keepdim=True)
    output = output / max_val_bone
    return output,max_val_bone