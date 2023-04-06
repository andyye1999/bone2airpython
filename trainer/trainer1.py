import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from util.utils import compute_STOI, compute_PESQ
from trainer.pipline import NetFeeder, Resynthesizer
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (mixture, clean, name) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            feeder = NetFeeder()

            self.optimizer.zero_grad()
            pred_mag, air_mag, pha= feeder(mixture, clean)
            # bone_normalized = mixture
            # bone_normalized = torch.squeeze(bone_normalized,dim=1)
            # pred_stft = torch.stft(bone_normalized, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
            # pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
            # pred_stft_real[:, 38:48, :] = pred_stft_real[:, 101:111, :]
            # pred_stft_real[:, 38:48, :] = pred_stft_real[:, 101:111, :]
            # pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
            # # pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
            # pred_mag[:, 36:71, :] = pred_mag[:, 1:36, :]
            # pred_mag[:, 71:106, :] = pred_mag[:, 1:36, :]
            # pred_mag[:, 106:141, :] = pred_mag[:, 1:36, :]
            # # input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
            #
            # air_normalized = clean
            # air_normalized = torch.squeeze(air_normalized, dim=1)
            # air_stft = torch.stft(air_normalized, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
            # air_stft_real, air_stft_imag = air_stft[:, :, :, 0], air_stft[:, :, :, 1]
            # air_stft_real[:, 38:48, :] = air_stft_real[:, 101:111, :]
            # air_stft_imag[:, 38:48, :] = air_stft_imag[:, 101:111, :]
            # air_mag = torch.sqrt(air_stft_real ** 2 + air_stft_imag ** 2 + 1e-6)
            # air_mag[:, 36:71, :] = air_mag[:, 1:36, :]
            # air_mag[:, 71:106, :] = air_mag[:, 1:36, :]
            # air_mag[:, 106:141, :] = air_mag[:, 1:36, :]
            # air_mag = torch.log(air_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
            # air_logmag = air_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)

            # enhanced = self.model(mixture)
            enhanced = self.model(pred_mag)
            loss = self.loss_function(air_mag, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
        print(epoch,loss_total / dl_len)
    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []

        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:

                # calculate maximum and minimum values of X for each batch
                # max_val_bone, _ = torch.max(torch.abs(mixture), dim=2, keepdim=True)
                # max_val_bone = torch.max(torch.abs(chunk))
                feeder = NetFeeder()
                resynthesizer = Resynthesizer()
                tmp = chunk
                pred_mag, air_mag, pha= feeder(chunk,tmp)
                # bone_normalized = chunk
                # # bone_normalized = mixture
                # bone_normalized = torch.squeeze(bone_normalized, dim=1)
                # pred_stft = torch.stft(bone_normalized, n_fft=320, hop_length=160, win_length=320)  # (Bs, F, T, 2)
                # pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
                # pred_stft_real[:, 38:48, :] = pred_stft_real[:, 101:111, :]
                # pred_stft_real[:, 38:48, :] = pred_stft_real[:, 101:111, :]
                # pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-6)
                # pred_mag[:, 36:71, :] = pred_mag[:, 1:36, :]
                # pred_mag[:, 71:106, :] = pred_mag[:, 1:36, :]
                # pred_mag[:, 106:141, :] = pred_mag[:, 1:36, :]
                # pred_mag = torch.log(pred_mag ** 2 + 1e-6)  # 将计算幅度谱改成计算对数幅度谱
                # input = pred_mag.permute(0, 2, 1)  # 指定维度新的位置 (Bs, T, F)
                enhance = self.model(pred_mag)
                output = resynthesizer(enhance,pha)
                # mag = enhance.permute(0, 2, 1)
                # mag = torch.exp(mag / 2)  # 将对数幅度谱转换成幅度谱
                # pha = torch.atan2(pred_stft_imag, pred_stft_real)  # ([16, 161, 103])
                # real = enhance * torch.cos(pha)
                # imag = enhance * torch.sin(pha)
                # model = torch.istft(torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1), n_fft=320,
                #                     hop_length=160,
                #                     win_length=320)
                # if torch.isnan(model).any():
                #     print("nan error")
                #     input("press enter ")
                output = torch.unsqueeze(output, 1)
                # enhanced_chunks.append(self.model(chunk).detach().cpu())
                enhanced_chunks.append(output)

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            # enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            if padded_length != 0:
                enhanced = enhanced[:, :, :-padded_length]
                mixture = mixture[:, :, :-padded_length]


            # enhanced = enhanced.reshape(-1).numpy()
            enhanced = enhanced.detach().cpu()
            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
            pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score
