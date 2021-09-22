import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from torch.cuda.amp import autocast
from nnAudio.Spectrogram import CQT1992v2, MelSpectrogram
import albumentations

from torch.fft import fft, rfft, ifft
from torchaudio.functional import bandpass_biquad

import torchaudio.transforms as T

from dropblock import DropBlock2D

import scipy


class V2Model(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3#, img_size=(128, 512)
        )
        self.n_features = self.model.fc.in_features
        self.embedding_size = 512
        # self.model.bn2 = nn.BatchNorm2d(1408, eps=0.001, momentum=0.2, affine=True, track_running_stats=True)
        self.model.fc = nn.Linear(self.n_features, self.cfg.target_size, bias=False)
        # torch.nn.init.normal_(self.model.head.weight, std=0.02)

        self.wave_transform = CQT1992v2(
            sr=2048, 
            fmin=20, 
            fmax=512, 
            hop_length=cfg.resolution,
            # norm=False
            # basis_norm=2
        ) #, bins_per_octave=12, filter_scale=16


    def forward(self, h_raw, l_raw, v_raw, m0, m1, m2, s0, s1, s2):
        with autocast():
            h = self.wave_transform(h_raw)
            h = (h - self.wave_transform(m0)) / self.wave_transform(s0)

            l = self.wave_transform(l_raw)
            l = (l - self.wave_transform(m1)) / self.wave_transform(s1)

            v = self.wave_transform(v_raw)
            v = (v - self.wave_transform(m2)) / self.wave_transform(s2)

            x = torch.stack([h, l, v], 1)

            # print(x.shape)

            # x = F.interpolate(x, (114, 257*self.cfg.image_width_factor))
            # x = F.interpolate(x, (128, 512))

            output = self.model(x)
            return output


class ViTModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        
        self.embedding_size = 512
        self.out_features = 768
        self.backbone = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3, img_size=(114, 514) #(57, 257)
        )
        
        self.neck = nn.Sequential(
                # nn.Dropout(0.3),
                nn.Linear(self.out_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        self.head = nn.Linear(self.embedding_size, self.cfg.target_size, bias=False)
        torch.nn.init.normal_(self.head.weight, std=0.02)

        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=cfg.resolution)

    def forward(self, h_raw, l_raw, v_raw):
        with autocast():
            h = self.wave_transform(h_raw) #+ 0.485
            # h = F.interpolate(h.unsqueeze(1), 224).squeeze(1)
            l = self.wave_transform(l_raw) #+ 0.456
            # l = F.interpolate(l.unsqueeze(1), 224).squeeze(1)
            v = self.wave_transform(v_raw) #+ 0.406
            # v = F.interpolate(v.unsqueeze(1), 224).squeeze(1)

            x = torch.stack([h, l, v], 1)

            x = F.interpolate(x, (114, 514))

            output = self.backbone.forward_features(x)
            output = output[0] + output[1]
            output = self.neck(output)

            return self.head(output)