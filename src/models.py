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


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

class V2Model(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg

        self.bn0 = nn.BatchNorm2d(3)
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3
        )
        self.n_features = self.model.classifier.in_features
        # self.model.bn2 = nn.BatchNorm2d(1408, eps=0.001, momentum=0.2, affine=True, track_running_stats=True)
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size, bias=False)
        # torch.nn.init.normal_(self.model.classifier.weight, std=0.02)

        # self.model.classifier = nn.Sequential(
        #         # nn.Dropout(0.1),
        #         nn.Linear(self.n_features, self.cfg.target_size, bias=False)
        #     )

        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)



    def forward(self, h_raw, l_raw, v_raw):
        with autocast():
            h = self.wave_transform(h_raw)
            l = self.wave_transform(l_raw)
            v = self.wave_transform(v_raw)

            x = torch.stack([h, l, v], 1)

            # x = self.bn0(x)
            output = self.model(x)
            return output


class ViTModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        
        self.embedding_size = 512
        self.out_features = 768
        self.backbone = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3, img_size=(57, 257)
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
            h = self.wave_transform(h_raw) #+ 0.485
            # h = F.interpolate(h.unsqueeze(1), 224).squeeze(1)
            l = self.wave_transform(l_raw) #+ 0.456
            # l = F.interpolate(l.unsqueeze(1), 224).squeeze(1)
            v = self.wave_transform(v_raw) #+ 0.406
            # v = F.interpolate(v.unsqueeze(1), 224).squeeze(1)

            x = torch.stack([h, l, v], 1)

            output = self.backbone.forward_features(x)
            output = output[0] + output[1]
            output = self.neck(output)

            return self.head(output)