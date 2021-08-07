import torch
import torch.nn as nn
import timm

from torch.cuda.amp import autocast

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

    def forward(self, x):
        with autocast():
            output = self.backbone.forward_features(x)
            output = output[0] + output[1]
            output = self.neck(output)

            return self.head(output)

