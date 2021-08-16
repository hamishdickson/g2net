import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torch.cuda.amp import autocast
from nnAudio.Spectrogram import CQT1992v2
import albumentations

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=1
        )
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, x):
        with autocast():
            output = self.model(x)
            return output

# class V2Model(nn.Module):
#     def __init__(self, cfg, pretrained=False):
#         super().__init__()
#         self.cfg = cfg
#         self.model = timm.create_model(
#             self.cfg.model_name, pretrained=pretrained, in_chans=3
#         )
#         self.n_features = self.model.classifier.in_features
#         self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)

#     def forward(self, x):
#         with autocast():
#             output = self.model(x)
#             return output


class V3Model(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        
        self.embedding_size = 512
        self.backbone = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3
        )
        self.out_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(self.out_features, self.cfg.target_size)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Linear(self.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        self.head = nn.Linear(self.embedding_size, self.cfg.target_size)
        torch.nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        with autocast():
            output = self.backbone.forward_features(x)
            output = self.global_pool(output)
            output = output[:,:,0,0]
            output = self.neck(output)

            return self.head(output)


class V2Model(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3
        )
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size, bias=False)

        # self.bn_0 = nn.BatchNorm1d(4096)
        # self.bn_1 = nn.BatchNorm1d(4096)
        # self.bn_2 = nn.BatchNorm1d(4096)

        # self.calibrate_0 = nn.Linear(1, 1)
        # self.calibrate_1 = nn.Linear(1, 1)
        # self.calibrate_2 = nn.Linear(1, 1)
        # self.calibrate_0 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        # self.calibrate_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        # self.calibrate_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)


    def forward(self, h, l, v):
        with autocast():
            # h = self.calibrate_0(h.unsqueeze(1))
            # h = h.squeeze(1)
            # l = self.calibrate_1(l.unsqueeze(1))
            # l = l.squeeze(1)
            # v = self.calibrate_2(v.unsqueeze(1))
            # v = v.squeeze(1)

            h = self.wave_transform(h)
            l = self.wave_transform(l)
            v = self.wave_transform(v)

            x = torch.stack([h, l, v], 1)
            output = self.model(x)
            return output

class ViTModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3, img_size=(57, 257)
        )
        self.embedding_size = 512
        self.out_features = 768
        # self.backbone.head = nn.Linear(self.out_features, self.cfg.target_size)

        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)

        self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        self.head = nn.Linear(self.embedding_size, self.cfg.target_size, bias=False)
        torch.nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, h, l, v):
        with autocast():
            h = self.wave_transform(h)
            l = self.wave_transform(l)
            v = self.wave_transform(v)

            x = torch.stack([h, l, v], 1)
            # print("here")
            output = self.backbone.forward_features(x)
            # print(output.shape)
            output = output[0] + output[1]
            output = self.neck(output)

            return self.head(output)

