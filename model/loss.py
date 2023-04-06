import torch
import numpy as np
import math
import auraloss
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss, apply_reduction
smallVal = np.finfo("float").eps  # To avoid divide by zero

class mag_loss(torch.nn.Module):
    def __init__(self):
        super(mag_loss, self).__init__()

    def forward(self, y_pred, y_true):
        # WINDOW = torch.sqrt(torch.hann_window(400, device=torch.device("cuda:0")) + 1e-8)
        # snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),
        #                 (torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        # snr_loss = 10 * torch.log10(snr + 1e-7)
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        pred_stft = torch.stft(y_pred, n_fft=320,hop_length=160,win_length=320)
        true_stft = torch.stft(y_true, n_fft=320,hop_length=160,win_length=320)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-6)
        # pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        # pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        # true_real_c = true_stft_real / (true_mag ** (2 / 3))
        # true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        # real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        # imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        loss = torch.mean((pred_mag  - true_mag ) ** 2)

        # loss = torch.log(real_loss + imag_loss + mag_loss + 1e-8) + snr_loss

        return loss

class copy_loss(torch.nn.Module):
    def __init__(self):
        super(copy_loss, self).__init__()

    def forward(self, y_pred, y_true):

        pred_mag1 = y_pred[:,0:65,:]
        pred_mag2 = y_pred[:,65:129,:]
        pred_mag3 = y_pred[:,129:193,:]
        pred_mag4 = y_pred[:,193:257,:]
        true_mag1 = y_true[:, 0:65, :]
        true_mag2 = y_true[:, 65:129, :]
        true_mag3 = y_true[:, 129:193, :]
        true_mag4 = y_true[:, 193:257, :]

        # pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        # pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        # true_real_c = true_stft_real / (true_mag ** (2 / 3))
        # true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        # real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        # imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        loss1 = torch.mean((pred_mag1  - true_mag1 ) ** 2)
        loss2 = torch.mean((pred_mag2 - true_mag2) ** 2)
        loss3 = torch.mean((pred_mag3 - true_mag3) ** 2)
        loss4 = torch.mean((pred_mag4 - true_mag4) ** 2)
        loss = 2*loss1 + 3*loss2 + 5*loss3 + 5*loss4
        # loss = torch.mean((pred_mag  - true_mag ) ** 2)

        # loss = torch.log(real_loss + imag_loss + mag_loss + 1e-8) + snr_loss

        return loss


def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

class logmag_loss(torch.nn.Module):
    def __init__(self):
        super(logmag_loss, self).__init__()

    def forward(self, y_air, y_enhance):
        # WINDOW = torch.sqrt(torch.hann_window(400, device=torch.device("cuda:0")) + 1e-8)
        # snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),
        #                 (torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        # snr_loss = 10 * torch.log10(snr + 1e-7)
        mubone = -9.981625
        sigmabone = 15.002887
        muair = -6.493551
        sigmaair = 15.735691
        max_val_enhance = torch.max(torch.abs(y_enhance), dim = -1, keepdim=True).values
        max_val_air = torch.max(torch.abs(y_air), dim = -1, keepdim=True).values
        # y_enhance_normalized = y_enhance / max_val_enhance
        # y_air_normalized = y_air / max_val_air
        y_enhance_normalized = y_enhance
        y_air_normalized = y_air
        y_enhance_normalized = torch.squeeze(y_enhance_normalized)
        y_air_normalized = torch.squeeze(y_air_normalized)
        enhance_stft = torch.stft(y_enhance_normalized, n_fft=320,hop_length=160,win_length=320)
        air_stft = torch.stft(y_air_normalized, n_fft=320,hop_length=160,win_length=320)
        enhance_stft_real, enhance_stft_imag = enhance_stft[:, :, :, 0], enhance_stft[:, :, :, 1]
        air_stft_real, air_stft_imag = air_stft[:, :, :, 0], air_stft[:, :, :, 1]
        enhance_mag = torch.sqrt(enhance_stft_real ** 2 + enhance_stft_real ** 2 + 1e-6)
        enhance_mag = torch.log(enhance_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # enhance_mag = (enhance_mag - mubone) / sigmabone
        air_mag = torch.sqrt(air_stft_real ** 2 + air_stft_imag ** 2 + 1e-6)
        air_mag = torch.log(air_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # air_mag = (air_mag - muair) / sigmaair
        # pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        # pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        # true_real_c = true_stft_real / (true_mag ** (2 / 3))
        # true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        # real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        # imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        loss = torch.mean((enhance_mag  - air_mag ) ** 2)

        # loss = torch.log(real_loss + imag_loss + mag_loss + 1e-8) + snr_loss

        return loss

class wavmag_loss(torch.nn.Module):
    def __init__(self):
        super(wavmag_loss, self).__init__()

    def forward(self, y_air, y_enhance):
        # WINDOW = torch.sqrt(torch.hann_window(400, device=torch.device("cuda:0")) + 1e-8)
        # snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),
        #                 (torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        # snr_loss = 10 * torch.log10(snr + 1e-7)
        # mubone = -9.981625
        # sigmabone = 15.002887
        # muair = -6.493551
        # sigmaair = 15.735691
        # max_val_enhance = torch.max(torch.abs(y_enhance), dim = -1, keepdim=True).values
        # max_val_air = torch.max(torch.abs(y_air), dim = -1, keepdim=True).values
        # y_enhance_normalized = y_enhance / max_val_enhance
        # y_air_normalized = y_air / max_val_air
        y_enhance_normalized = y_enhance
        y_air_normalized = y_air
        y_enhance_normalized = torch.squeeze(y_enhance_normalized)
        y_air_normalized = torch.squeeze(y_air_normalized)
        enhance_stft = torch.stft(y_enhance_normalized, n_fft=320,hop_length=160,win_length=320)
        air_stft = torch.stft(y_air_normalized, n_fft=320,hop_length=160,win_length=320)
        enhance_stft_real, enhance_stft_imag = enhance_stft[:, :, :, 0], enhance_stft[:, :, :, 1]
        air_stft_real, air_stft_imag = air_stft[:, :, :, 0], air_stft[:, :, :, 1]
        enhance_mag = torch.sqrt(enhance_stft_real ** 2 + enhance_stft_real ** 2 + 1e-6)
        enhance_mag = torch.log(enhance_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # enhance_mag = (enhance_mag - mubone) / sigmabone
        air_mag = torch.sqrt(air_stft_real ** 2 + air_stft_imag ** 2 + 1e-6)
        air_mag = torch.log(air_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
        # air_mag = (air_mag - muair) / sigmaair
        # pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        # pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        # true_real_c = true_stft_real / (true_mag ** (2 / 3))
        # true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        # real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        # imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        lossmag = torch.mean((enhance_mag  - air_mag ) ** 2)
        losswav = torch.mean((y_air - y_enhance) ** 2)
        loss = lossmag + losswav
        # loss = torch.log(real_loss + imag_loss + mag_loss + 1e-8) + snr_loss

        return loss.mean()



class multiloss(torch.nn.Module):
    def __init__(self):
        super(multiloss, self).__init__()

    def forward(self, input, target):
        mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[512, 256, 1024], hop_sizes=[100, 50, 120],
                                                       win_lengths=[400, 200, 600])
        loss = mrstft(input, target)
        return loss

class stftloss(torch.nn.Module):
    def __init__(self):
        super(stftloss, self).__init__()

    def forward(self, input, target):
        mrstft = auraloss.freq.STFTLoss(fft_size=400, hop_size=200,
                                                       win_length=400)
        loss = mrstft(input, target)
        return loss

class dpcrnloss(torch.nn.Module):
    def __init__(self):
        super(dpcrnloss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_stft = torch.stft(y_pred, 400, 200, win_length=400)
        true_stft = torch.stft(y_true, 400, 200, win_length=400)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)

        return real_loss + imag_loss + mag_loss


class realimagloss(torch.nn.Module):
    def __init__(self):
        super(realimagloss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_stft = torch.stft(y_pred, 400, 200, win_length=400)
        true_stft = torch.stft(y_true, 400, 200, win_length=400)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        # mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)

        return real_loss + imag_loss

class dccrnloss(torch.nn.Module):
    def __init__(self):
        super(dccrnloss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_stft = torch.stft(y_pred, 400, 200, win_length=400)
        true_stft = torch.stft(y_true, 400, 200, win_length=400)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        # pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        # true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        # pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        # pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        # true_real_c = true_stft_real / (true_mag ** (2 / 3))
        # true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_stft_real - true_stft_real) ** 2)
        imag_loss = torch.mean((pred_stft_imag - true_stft_imag) ** 2)
        # mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)

        return real_loss + imag_loss

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

class sisnrloss(torch.nn.Module):
    def __init__(self):
        super(sisnrloss, self).__init__()

    def forward(self, input, target):
        return -(si_snr(input, target))

class dpcrntestloss(torch.nn.Module):
    def __init__(self):
        super(dpcrntestloss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_stft = torch.stft(y_pred, 400, 200, win_length=400)
        true_stft = torch.stft(y_true, 400, 200, win_length=400)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)

        return real_loss + imag_loss + mag_loss, real_loss,imag_loss,mag_loss

class dpcrnmultiloss(torch.nn.Module):
    def __init__(self):
        super(dpcrnmultiloss, self).__init__()

    def forward(self, y_pred, y_true):
        pred_stft = torch.stft(y_pred, 400, 200, win_length=400)
        true_stft = torch.stft(y_true, 400, 200, win_length=400)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)
        mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[512, 256, 1024], hop_sizes=[200, 50, 120],
                                                       win_lengths=[400, 200, 600],)
        mrloss = mrstft(y_pred, y_true)
        return real_loss + imag_loss + mag_loss + mrloss

class generator_loss(torch.nn.Module):
    def __init__(self):
        super(generator_loss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.relu = torch.nn.ReLU()

    def forward(self, reference_embeddings, enhanced_embeddings):
        ftr_loss = 0
        for scale in range(len(reference_embeddings)):  # across scales
            for layer in range(1, len(reference_embeddings[scale]) - 1):  # across layers
                a = reference_embeddings[scale][layer]
                b = enhanced_embeddings[scale][layer]
                # ftr_loss += self.l1(a, b) / (len(reference_embeddings[scale]) - 2)
                ftr_loss += self.l1(a, b) / (torch.mean(torch.abs(a)) * (len(reference_embeddings[scale]) - 2))  # normalized
        ftr_loss /= len(reference_embeddings)

        # loss_adv_gen
        adv_loss = 0
        for scale in range(len(enhanced_embeddings)):  # across embeddings
            certainties = enhanced_embeddings[scale][-1]
            adv_loss += self.relu(1 - certainties).mean()  # across time
        adv_loss /= len(enhanced_embeddings)


        # gen_loss = adv_loss + 100 * ftr_loss


        return adv_loss,ftr_loss

class discriminator_loss(torch.nn.Module):
    def __init__(self):
        super(discriminator_loss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.relu = torch.nn.ReLU()

    def forward(self, reference_embeddings, enhanced_embeddings):
        # valid_loss
        adv_loss_valid = 0
        for scale in range(len(reference_embeddings)):  # across embeddings
            certainties = reference_embeddings[scale][-1]
            adv_loss_valid += self.relu(1 - certainties).mean()  # across time
        adv_loss_valid /= len(reference_embeddings)

        # fake_loss
        adv_loss_fake = 0
        for scale in range(len(enhanced_embeddings)):  # across embeddings
            certainties = enhanced_embeddings[scale][-1]
            adv_loss_fake += self.relu(1 + certainties).mean()  # across time
        adv_loss_fake /= len(enhanced_embeddings)

        # loss to backprop on
        dis_loss = adv_loss_valid + adv_loss_fake
        return dis_loss

class STFTLossDDP(STFTLoss):

    def forward(self, x, y):
        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb.to(x_mag.device), x_mag)
            y_mag = torch.matmul(self.fb.to(y_mag.device), y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0

        # combine loss terms
        loss = (self.w_sc * sc_loss) + (self.w_log_mag * mag_loss) + (self.w_lin_mag * lin_loss)
        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_loss, mag_loss


class MRSTFTLossDDP(MultiResolutionSTFTLoss):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 **kwargs):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLossDDP(fs,
                                             ss,
                                             wl,
                                             window,
                                             w_sc,
                                             w_log_mag,
                                             w_lin_mag,
                                             w_phs,
                                             sample_rate,
                                             scale,
                                             n_bins,
                                             scale_invariance,
                                             **kwargs)]

class tunetloss(torch.nn.Module):
    def __init__(self):
        super(tunetloss, self).__init__()
        self.time_loss = torch.nn.MSELoss()
        self.freq_loss = MRSTFTLossDDP(n_bins=64, sample_rate=16000, device="cpu", scale='mel')
    def forward(self, x, y):
        loss = self.freq_loss(x, y) + self.time_loss(x, y) * 10000
        # loss = self.freq_loss(x, y) + self.time_loss(x, y) * 2

        return loss

class cntloss(torch.nn.Module):
    def __init__(self):
        super(cntloss, self).__init__()
        self.time_loss = torch.nn.L1Loss()
        self.freq_loss = MRSTFTLossDDP(n_bins=64, sample_rate=16000, device="cpu", scale='mel')

    def forward(self, x, y):
        # loss = self.freq_loss(x, y) + self.time_loss(x, y) * 10000
        loss = self.freq_loss(x, y) * 0.5 + self.time_loss(x, y)

        return loss

