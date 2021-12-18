import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
import wandb
from tqdm import tqdm
from model import HIFIGenerator
import sys
from data_preprocessing import LJSpeechCollator, LJSpeechDataset
from data_preprocessing import MelSpectrogramConfig, MelSpectrogram


device = 'cpu'
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=10, collate_fn=LJSpeechCollator(device))
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
model = HIFIGenerator().to(device)
loss_function = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
sr = 22050

model.train()
waveform, melspecs = next(iter(dataloader))
for i in tqdm(range(5000)):
    prediction = model(melspecs)
    prediction = prediction[:, :, :waveform.shape[1]].squeeze(1)
    mel_pred = featurizer(prediction)
    loss = loss_function(mel_pred, melspecs)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()