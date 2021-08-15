import torch
import torch.nn as nn
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

class V2Model(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3
        )
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, x):
        with autocast():
            output = self.model(x)
            return output


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




# class V2Model(nn.Module):
#     def __init__(self, cfg, pretrained=False):
#         super().__init__()
#         self.cfg = cfg
#         self.model = timm.create_model(
#             self.cfg.model_name, pretrained=pretrained, in_chans=3
#         )
#         self.n_features = self.model.classifier.in_features
#         self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)

#         self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)

#     def forward(self, h, l, v, aug=False):
#         with autocast():
#             h = self.wave_transform(h)
#             l = self.wave_transform(l)
#             v = self.wave_transform(v)
#             if aug:
#                 h = albumentations.CoarseDropout(p=0.2)(image=h)
#                 l = albumentations.CoarseDropout(p=0.2)(image=l)
#                 v = albumentations.CoarseDropout(p=0.2)(image=v)

#             x = torch.stack([h, l, v], 1)
#             output = self.model(x)
#             return output

class ViTModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        
        self.embedding_size = 512
        self.backbone = timm.create_model(
            self.cfg.model_name, pretrained=pretrained, in_chans=3, img_size=(57, 257)
        )
        self.out_features = 768
        self.backbone.classifier = nn.Linear(self.out_features, self.cfg.target_size)

        self.neck = nn.Sequential(
                # nn.Dropout(0.3),
                nn.Linear(self.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        self.head = nn.Linear(self.embedding_size, self.cfg.target_size)
        torch.nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        with autocast():
            output = self.backbone.forward_features(x)
            output = output[0] + output[1]
            output = self.neck(output)

            return self.head(output)

