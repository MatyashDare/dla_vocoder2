import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
import wandb
from tqdm import tqdm
from model import HIFIGenerator, MPD, MSD
import sys
from data_preprocessing import LJSpeechCollator, LJSpeechDataset
from data_preprocessing import MelSpectrogramConfig, MelSpectrogram


device = 'cpu'
dataloader = DataLoader(LJSpeechDataset('.', 8192), batch_size=16, collate_fn=LJSpeechCollator(device))
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
model = HIFIGenerator().to(device)
mpd = MPD().to(device)
msd = MSD().to(device)
mae = nn.L1Loss()
mse = nn.MSELoss()
optimizer_g = optim.AdamW(model.parameters(), 2e-4, (0.8, 0.99), weight_decay=0.01)
optimizer_msd = optim.AdamW(msd.parameters(), 2e-4, (0.8, 0.99), weight_decay=0.01)
optimizer_mpd = optim.AdamW(mpd.parameters(), 2e-4, (0.8, 0.99), weight_decay=0.01)


test_mels = []
test_audios_wandb = []
for i in range(1, 3):
    path = f'test_audio/audio_{i}.wav'
    test_audio, _ = torchaudio.load(path)
    test_mels.append(featurizer(test_audio.to(device)))
    test_audios_wandb.append(wandb.Audio(test_audio.numpy()[0], 22050))

with wandb.init(project="tts_check", name="check") as run:
    for n in range(100):
        model.train()
        i = 0
        for batch in tqdm(dataloader):
            waveform, melspecs = batch
            prediction = model(melspecs)
            prediction = prediction[:, :, :waveform.shape[1]].squeeze(1)
            prediction_ = prediction.detach()
            mel_pred = featurizer(prediction)
            # mel loss
            mel_loss = mae(mel_pred, melspecs)

            # feature loss
            msd_fake_ans, msd_fake_fs = msd(prediction)
            msd_real_ans, msd_real_fs = msd(waveform)

            mpd_fake_ans, mpd_fake_fs = mpd(prediction)
            mpd_real_ans, mpd_real_fs = mpd(waveform)

            msd_feature_loss = 0
            for fake_f, real_f in zip(msd_fake_fs, msd_real_fs):
                msd_feature_loss += mse(fake_f, real_f)
            msd_feature_loss /= len(msd_fake_fs)

            mpd_feature_loss = 0
            for fake_f, real_f in zip(mpd_fake_fs, mpd_real_fs):
                mpd_feature_loss += mse(fake_f, real_f)
            mpd_feature_loss /= len(mpd_fake_fs)

            feature_loss = msd_feature_loss + mpd_feature_loss

            # adv loss
            mpd_gen_adv_loss = 0
            for a in mpd_fake_ans:
                target = torch.ones(a.shape)
                mpd_gen_adv_loss += mse(a, target)
            mpd_gen_adv_loss /= len(mpd_fake_ans)

            msd_gen_adv_loss = 0
            for a in msd_fake_ans:
                target = torch.ones(a.shape)
                msd_gen_adv_loss += mse(a, target)
            msd_gen_adv_loss /= len(msd_fake_ans)

            gen_adv_loss = mpd_gen_adv_loss + msd_gen_adv_loss
            gen_total_loss = 45 * mel_loss + 2 * feature_loss + gen_adv_loss


            optimizer_g.zero_grad()
            gen_total_loss.backward()
            optimizer_g.step()

            # discr

            mpd_fake_ans, _ = mpd(prediction_)
            mpd_real_ans, _ = mpd(waveform)

            mpd_disc_adv_loss = 0
            for f, g in zip(mpd_fake_ans, mpd_real_ans):
                real_target = torch.ones(g.shape)
                fake_target = torch.zeros(f.shape)
                mpd_disc_adv_loss += 0.5 * (mse(g, real_target) + mse(f, fake_target))
            mpd_disc_adv_loss /= len(mpd_fake_ans)

            optimizer_mpd.zero_grad()
            mpd_disc_adv_loss.backward()
            optimizer_mpd.step()

            msd_fake_ans, _ = msd(prediction_)
            msd_real_ans, _ = msd(waveform)

            msd_disc_adv_loss = 0
            for f, g in zip(msd_fake_ans, msd_real_ans):
                real_target = torch.ones(g.shape)
                fake_target = torch.zeros(f.shape)
                msd_disc_adv_loss += 0.5 * (mse(g, real_target) + mse(f, fake_target))
            msd_disc_adv_loss /= len(msd_fake_ans)

            optimizer_msd.zero_grad()
            msd_disc_adv_loss.backward()
            optimizer_msd.step()
            if i % 10 == 0:
                run.log({"gen_total_loss" : gen_total_loss}, step=n * len(dataloader) + i)
                run.log({"feature_loss" : feature_loss}, step=n * len(dataloader) + i)
                run.log({"gen_adv_loss" : gen_adv_loss}, step=n * len(dataloader) + i)
                run.log({"mpd_disc_adv_loss" : mpd_disc_adv_loss}, step=n * len(dataloader) + i)
                run.log({"msd_disc_adv_loss" : msd_disc_adv_loss}, step=n * len(dataloader) + i)
            if i % 400 == 0:
                run.log({"Predicted audio" : wandb.Audio(prediction[0].detach().cpu().numpy(), 22050)}, step=n * len(dataloader) + i)
                run.log({"Real Audio" : wandb.Audio(waveform[0].detach().cpu().numpy(), 22050)}, step=n * len(dataloader) + i)

                run.log({"Predicted melspec" : wandb.Image(mel_pred[0].detach().cpu().numpy())}, step=n * len(dataloader) + i)
                run.log({"Real melspec" : wandb.Image(melspecs[0].detach().cpu().numpy())}, step=n * len(dataloader) + i)
            i += 1
        model.eval()
        predicted_audios = []
        for mel in test_mels:
            with torch.no_grad():
                prediction = model(mel).squeeze(1)[0]
            predicted_audios.append(wandb.Audio(prediction.detach().cpu().numpy(), 22050))
        run.log({"TEST Predicted Audio" : predicted_audios}, step=(n + 1) * len(dataloader))
        run.log({"TEST Real Audio" : test_audios_wandb}, step=(n + 1) * len(dataloader))
