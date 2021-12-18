import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

PADDING_MODE = 'same'
RELUSLOPE = 0.1


class Block(nn.Module):
    def __init__(self, c_in, kernel_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=3, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=5, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
        )

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


class Resblocks(nn.Module):
    def __init__(self, c_in, resblocks_kernels):
        super().__init__()
        self.blocks = nn.ModuleList([Block(c_in, kernel_size) for kernel_size in resblocks_kernels])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class HIFIGenerator(nn.Module):
    def __init__(self,):
        super().__init__()
        first_channels = 512
        self.preconv = nn.Sequential(nn.Conv1d(80, first_channels, kernel_size=7, padding=PADDING_MODE), nn.LeakyReLU(RELUSLOPE))
        self.transposes = nn.ModuleList([])
        self.resblocks = nn.ModuleList([])
        kernels = [16, 16, 4, 4]
        resblocks_kernels = [3, 7, 11]
        c = first_channels
        for i in range(len(kernels)):
            self.transposes.append(nn.ConvTranspose1d(c, c // 2, kernel_size=kernels[i], stride=kernels[i] // 2, padding=kernels[i] // 4))
            self.resblocks.append(nn.Sequential(Resblocks(c // 2, resblocks_kernels), nn.LeakyReLU(RELUSLOPE)))
            c = c // 2
        self.postconv = nn.Sequential(nn.Conv1d(c, 1, kernel_size=7, padding=PADDING_MODE), nn.Tanh())

    def forward(self, x):
        x = self.preconv(x)
        for i in range(len(self.transposes)):
            x = self.transposes[i](x)
            x = self.resblocks[i](x)
        return self.postconv(x)


class PDiscr(torch.nn.Module):
    def __init__(self, period, norm_type='weight'):
        super(PDiscr, self).__init__()
        if norm_type == 'weight':
            norm = weight_norm
        elif norm_type == 'spectral':
            norm = spectral_norm
        self.period = period
        self.model = nn.ModuleList([nn.Sequential(norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0))), nn.LeakyReLU(RELUSLOPE)),
            norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))])

    def forward(self, x):
        batch_size, time = x.shape
        x = x.unsqueeze(1)
        if time % self.period != 0: # pad first
            padding_to_add = (time // self.period) * self.period - time
            x = F.pad(x, (0, padding_to_add), "reflect")
        x = x.view(batch_size, 1, x.shape[2] // self.period, self.period)
        fmaps = []
        for layer in self.model:
            x = layer(x)
            fmaps.append(x)
        return torch.flatten(x, 1, -1), fmaps


class MPD(torch.nn.Module):
    def __init__(self):
        super(MPD, self).__init__()
        self.discrs = nn.ModuleList([
            PDiscr(2), PDiscr(3), PDiscr(5), PDiscr(7), PDiscr(11),
        ])

    def forward(self, x):
        all_ans = []
        all_fmaps = []
        for i, d in enumerate(self.discrs):
            ans, fmaps = d(x)
            all_ans.append(ans)
            all_fmaps.extend(fmaps)
        return all_ans, all_fmaps


class SDiscr(torch.nn.Module):
    def __init__(self, norm_type='weight'):
        super(SDiscr, self).__init__()
        if norm_type == 'weight':
            norm = weight_norm
        elif norm_type == 'spectral':
            norm = spectral_norm
        self.model = nn.ModuleList([
            nn.Sequential(norm(nn.Conv1d(1, 128, 15, stride=1, padding=7)), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv1d(128, 256, 41, stride=2, groups=16, padding=20)), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20)), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv1d(512, 1024, 41, stride=4, groups=16, padding=20)), nn.LeakyReLU(RELUSLOPE)),
            nn.Sequential(norm(nn.Conv1d(1024, 1024, 41, stride=1, groups=16, padding=20)), nn.LeakyReLU(RELUSLOPE)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2))])

    def forward(self, x):
        fmaps = []
        for layer in self.model:
            x = layer(x)
            fmaps.append(x)
        return torch.flatten(x, 1, -1), fmaps

class MSD(torch.nn.Module):
    def __init__(self):
        super(MSD, self).__init__()
        self.discrs = nn.ModuleList([
            SDiscr(norm_type='spectral'),
            SDiscr(norm_type='weight'),
            SDiscr(norm_type='weight'),
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        x = x.unsqueeze(1)
        all_ans = []
        all_fmaps = []
        for i, d in enumerate(self.discrs):
            if i != 0:
                x = self.pooling(x)
            ans, fmaps = d(x)
            all_ans.append(ans)
            all_fmaps.extend(fmaps)
        return all_ans, all_fmaps

