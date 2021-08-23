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


class V2Model(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3
        )
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size, bias=False)
        # torch.nn.init.normal_(self.model.classifier.weight, std=0.02)

        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)


    # def whiten(self, signal):
    #     hann = torch.hann_window(signal.shape[1], periodic=True, dtype=float).cuda().half()
    #     # print(hann.shape, signal.shape)
    #     spec = fft(signal* hann)
    #     mag = torch.sqrt(torch.real(spec*torch.conj(spec))) 

    #     return torch.real(ifft(spec/mag)) * np.sqrt(signal.shape[1]/2)


    def forward(self, h_raw, l_raw, v_raw):
        with autocast():

            ## nans out
            # h_raw = self.whiten(h_raw)
            # l_raw = self.whiten(l_raw)
            # v_raw = self.whiten(v_raw)

            # ## slow!!
            # h_raw = bandpass_biquad(h_raw, 2048, (512 + 20) / 2, (512 - 20) / (512 + 20))
            # l_raw = bandpass_biquad(l_raw, 2048, (512 + 20) / 2, (512 - 20) / (512 + 20))
            # v_raw = bandpass_biquad(v_raw, 2048, (512 + 20) / 2, (512 - 20) / (512 + 20))

            h = self.wave_transform(h_raw) #+ 0.485
            # h = F.interpolate(h.unsqueeze(1), 257).squeeze(1)
            l = self.wave_transform(l_raw) #+ 0.456
            # l = F.interpolate(l.unsqueeze(1), 257).squeeze(1)
            v = self.wave_transform(v_raw) #+ 0.406
            # v = F.interpolate(v.unsqueeze(1), 257).squeeze(1)

            x = torch.stack([h, l, v], 1)
            output = self.model(x)
            return output


class ViTModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        
        self.embedding_size = 512
        self.out_features = 768
        self.backbone = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3 #, img_size=(57, 257)
        )
        
        self.neck = nn.Sequential(
                # nn.Dropout(0.3),
                nn.Linear(self.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        self.head = nn.Linear(self.embedding_size, self.cfg.target_size, bias=False)
        torch.nn.init.normal_(self.head.weight, std=0.02)

        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)

    def forward(self, h_raw, l_raw, v_raw):
        with autocast():
            h = self.wave_transform(h_raw) + 0.485
            h = F.interpolate(h.unsqueeze(1), 224).squeeze(1)
            l = self.wave_transform(l_raw) + 0.456
            l = F.interpolate(l.unsqueeze(1), 224).squeeze(1)
            v = self.wave_transform(v_raw) + 0.406
            v = F.interpolate(v.unsqueeze(1), 224).squeeze(1)

            x = torch.stack([h, l, v], 1)

            output = self.backbone.forward_features(x)
            output = output[0] + output[1]
            output = self.neck(output)

            return self.head(output)