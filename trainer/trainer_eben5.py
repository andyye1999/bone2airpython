import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer_eben5 import BaseTrainer
from util.utils import compute_STOI, compute_PESQ
from src.generator import GeneratorEBEN
from src.discriminator import DiscriminatorEBENMultiScales
plt.switch_backend('agg')

WARMUP_ITERATIONS = 10000


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
        for i, (mixture, clean, name, max_bone, max_air) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            max_bone = max_bone.to(self.device)
            max_air = max_air.to(self.device)

            mixture = self.model.cut_tensor(mixture)
            clean = self.model.cut_tensor(clean)
            train_gen = self.step % 2 == 0
            train_dsc = self.step % 2 == 1
            # train_dsc = self.step % 2 == 1 self.step >= WARMUP_ITERATIONS
            # train generator
            self.optimizer.zero_grad()
            enhanced_speech, decomposed_enhanced_speech = self.model(mixture)

            if train_gen:
                # gen_loss = cnt_loss = self.loss1(clean, enhanced_speech)
                # self.writer.add_scalar(f"Train/cntLoss", cnt_loss.item() , self.step)

                decomposed_reference_speech = self.model.pqmf.forward(clean, 'analysis')
                enhanced_embeddings = self.discriminator(bands=decomposed_enhanced_speech[:, 1:, :],
                                                             audio=enhanced_speech)
                reference_embeddings = self.discriminator(bands=decomposed_reference_speech[:, 1:, :],
                                                              audio=clean)

                # gen_loss = self.loss_function(reference_embeddings, enhanced_embeddings)
                adv_loss, ftr_loss = self.loss_function(reference_embeddings, enhanced_embeddings)
                lamda = self.compute_ema_lambda_adaptive(ftr_loss,adv_loss)
                gen_loss = adv_loss + lamda * ftr_loss
                self.writer.add_scalar(f"Train/genLoss", gen_loss.item(), self.step)
                self.writer.add_scalar(f"Train/adv_loss", adv_loss.item(), self.step)
                self.writer.add_scalar(f"Train/ftr_loss", ftr_loss.item(), self.step)

                gen_loss_total += gen_loss.item()
                # enhanced = self.model(mixture)
                # loss = self.loss_function(clean, enhanced)
                gen_loss.backward()
                self.optimizer.step()
            # train discriminator
            if train_dsc:
                self.optimizer2.zero_grad()
                # enhanced_speech, decomposed_enhanced_speech = self.model(mixture)

                decomposed_reference_speech = self.model.pqmf.forward(clean, 'analysis')
                enhanced_speech_detach = enhanced_speech.detach()
                decomposed_enhanced_speech_detach = decomposed_enhanced_speech.detach()
                enhanced_embeddings = self.discriminator(bands=decomposed_enhanced_speech_detach[:, 1:, :],
                                                         audio=enhanced_speech_detach)
                reference_embeddings = self.discriminator(bands=decomposed_reference_speech[:, 1:, :],
                audio=clean)
                dis_loss = self.loss2(reference_embeddings, enhanced_embeddings)
                dis_loss_total += dis_loss.item()
                dis_loss.backward()
                self.optimizer2.step()
                self.writer.add_scalar(f"Train/disLoss", dis_loss.item(), self.step)

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

        for i, (mixture, clean, name, max_bone, max_air) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]
            clean = clean.to(self.device)
            max_bone = max_bone.to(self.device)
            max_air = max_air.to(self.device)
            mixture = self.model.cut_tensor(mixture)
            clean = self.model.cut_tensor(clean)
            enhanced_speech, _ = self.model(mixture)
            # # The input of the model should be fixed length.
            # if mixture.size(-1) % sample_length != 0:
            #     padded_length = sample_length - (mixture.size(-1) % sample_length)
            #     mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)
            #
            # assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            # mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))
            #
            # enhanced_chunks = []
            # for chunk in mixture_chunks:
            #     enhanced_chunks.append(self.model(chunk).detach().cpu())
            #
            # enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            # # enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            # if padded_length != 0:
            #     enhanced = enhanced[:, :, :-padded_length]
            #     mixture = mixture[:, :, :-padded_length]

            enhanced_speech = enhanced_speech * max_bone
            clean = clean * max_air
            mixture = mixture * max_bone
            enhanced = enhanced_speech.detach().cpu()
            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.cpu().numpy().reshape(-1)
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
