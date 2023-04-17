import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.base_trainer_soundstream3 import BaseTrainer
from util.utils import compute_STOI, compute_PESQ,compute_PESQ8k
from src.generator import GeneratorEBEN
from src.discriminator import DiscriminatorEBENMultiScales
plt.switch_backend('agg')

WARMUP_ITERATIONS = 15000


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
        gen_loss_total = 0
        dis_loss_total = 0
        for i, (mixture, filename, max_bone) in enumerate(self.train_data_loader):
            clean = mixture
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            max_bone = max_bone.to(self.device)
            # max_air = max_air.to(self.device)

            # mixture = self.model.cut_tensor(mixture)
            # clean = self.model.cut_tensor(clean)
            train_gen = self.step % 2 == 0
            train_dsc = self.step % 2 == 1
            # train_dsc = self.step % 2 == 1 self.step >= WARMUP_ITERATIONS
            # train generator
            self.optimizer.zero_grad()
            enhanced_speech = self.model(mixture)

            if train_gen:
                loss_rec_time = F.l1_loss(enhanced_speech, mixture, reduction="mean")
                loss_rec_mel = self.loss2(enhanced_speech, mixture)
                loss_G = 10 * loss_rec_time + loss_rec_mel

                logits_D_fake, features_D_fake = self.discriminator(enhanced_speech)
                logits_D_real, features_D_real = self.discriminator(mixture)

                # gen_loss = self.loss_function(reference_embeddings, enhanced_embeddings)
                loss_fm = 0
                loss_adv = 0
                for i, scale in enumerate(logits_D_fake):
                    loss_adv += F.relu(1-scale).mean()
                loss_adv /= len(logits_D_fake)
                for i in range(len(features_D_fake)):
                    for j in range(len(features_D_fake[0])):
                        loss_fm += F.l1_loss(features_D_fake[i][j], features_D_real[i][j].detach(), reduction="mean") / \
                                   (features_D_real[i][j].detach().abs().mean() * (len(features_D_fake[0])-1))
                loss_fm /= len(features_D_fake)
                self.writer.add_scalar(f"Train/loss_fm", loss_fm, self.step)
                self.writer.add_scalar(f"Train/loss_adv", loss_adv, self.step)
                lamda = self.compute_ema_lambda_adaptive(loss_fm,loss_adv)
                loss_G += lamda * loss_fm + loss_adv

                self.writer.add_scalar(f"Train/loss_GLoss", loss_G.item(), self.step)
                self.writer.add_scalar(f"Train/loss_rec_time", loss_rec_time.item(), self.step)
                # self.writer.add_scalar(f"Train/commit_loss", commit_loss.item(), self.step)

                self.writer.add_scalar(f"Train/loss_rec_mel", loss_rec_mel.item(), self.step)

                gen_loss_total += loss_G.item()

                # enhanced = self.model(mixture)
                # loss = self.loss_function(clean, enhanced)
                loss_G.backward()
                self.optimizer.step()
            # train discriminator
            if train_dsc:
                self.optimizer2.zero_grad()
                # enhanced_speech, decomposed_enhanced_speech = self.model(mixture)

                enhanced_speech_detach = enhanced_speech.detach()

                logits_D_fake, features_D_fake = self.discriminator(enhanced_speech_detach)
                logits_D_real, features_D_real = self.discriminator(mixture)
                loss_D = 0
                loss_D1 = 0
                loss_D2 = 0
                for i, scale in enumerate(logits_D_fake):
                    loss_D1 += F.relu(1 + scale).mean()
                loss_D1 /= len(logits_D_fake)

                for i, scale in enumerate(logits_D_real):
                    loss_D2 += F.relu(1 - scale).mean()
                loss_D2 /= len(logits_D_real)
                loss_D = loss_D1 + loss_D2
                dis_loss_total += loss_D.item()
                loss_D.backward()
                self.optimizer2.step()
                self.writer.add_scalar(f"Train/disLoss", loss_D.item(), self.step)
                self.writer.add_scalar(f"Train/loss_D1", loss_D1, self.step)
                self.writer.add_scalar(f"Train/loss_D2", loss_D2, self.step)

            self.step += 1

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/gentotalLoss", gen_loss_total / dl_len, epoch)
        self.writer.add_scalar(f"Train/distotalLoss", dis_loss_total / dl_len, epoch)
        # self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

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

        for i, (mixture, name, max_bone) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0
            clean = mixture
            mixture = mixture.to(self.device)  # [1, 1, T]
            clean = clean.to(self.device)
            max_bone = max_bone.to(self.device)
            # max_air = max_air.to(self.device)
            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                output= self.model(chunk)
                output = output * max_bone
                # enhanced_chunks.append(self.model(chunk).detach().cpu())
                enhanced_chunks.append(output)
            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            if padded_length != 0:
                enhanced = enhanced[:, :, :-padded_length]
                mixture = mixture[:, :, :-padded_length]

            clean = clean * max_bone
            mixture = mixture * max_bone
            enhanced = enhanced.detach().cpu()
            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.cpu().numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=8000)

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
                    librosa.display.waveplot(y, sr=8000, ax=ax[j])
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
            stoi_c_n.append(compute_STOI(clean, mixture, sr=8000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=8000))
            pesq_c_n.append(compute_PESQ8k(clean, mixture, sr=8000))
            pesq_c_e.append(compute_PESQ8k(clean, enhanced, sr=8000))

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
