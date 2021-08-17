import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from nnAudio.Spectrogram import CQT1992v2
import albumentations
import random


class TrainDataset(Dataset):
    def __init__(self, CFG, df, transform=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["target"].values
        # TRYME worth playing with these values
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)
        self.transform = transform
        self.detector = CFG.detector
        self.calibration = CFG.calibration

    def __len__(self):
        return len(self.df)


    def apply_qtransform(self, waves, qtransform):
        
        w = waves[self.detector]
        w = w / self.calibration
        w = torch.from_numpy(w).float()
        image = qtransform(w)

        return image

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        image = self.apply_qtransform(waves, self.wave_transform)
        if self.transform:
            image = image.squeeze().numpy()
            image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx]).float()
        return image, label



# class ThreeTrainDataset(Dataset):
#     def __init__(self, CFG, df, transform=None):
#         self.df = df
#         self.file_names = df["file_path"].values
#         self.labels = df["target"].values
#         # TRYME worth playing with these values
#         self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def apply_qtransform(self, waves, qtransform):
#         w0 = waves[0]
#         w0 = w0 / 4.615211621383077e-20 #np.max(w0)
#         # w0 = w0 + (1.7259105809493188e-20 / 2)
#         # if self.transform:
#         #     w0 = w0.squeeze()
#         #     w0 = self.transform(w0, sample_rate=2048)
#         # w0 = np.clip(w0, -1, 1)
#         w0 = torch.from_numpy(w0).float()
#         i0 = qtransform(w0)

#         w1 = waves[1]
#         w1 = w1 / 4.1438353591025024e-20 #np.max(w1)
#         # w1 = w1 + (1.7262760441525487e-20 / 2)
#         # if self.transform:
#         #     w1 = w1.squeeze()
#         #     w1 = self.transform(w1, sample_rate=2048)
#         # w1 = np.clip(w1, -1, 1)
#         w1 = torch.from_numpy(w1).float()
#         i1 = qtransform(w1)

#         w2 = waves[2]
#         w2 = w2 / 6e-20 #1.1161063663761836e-20 #np.max(w2)
#         # w2 = w2 + (4.276536160007436e-21 / 2)
#         # if self.transform:
#             # w2 = w2.squeeze()
#             # w2 = self.transform(w2, sample_rate=2048)
#         # w2 = np.clip(w2, -1, 1)
#         w2 = torch.from_numpy(w2).float()
#         i2 = qtransform(w2)

#         image = np.vstack([i0, i1, i2])

#         return image


#     def __getitem__(self, idx):
#         file_path = self.file_names[idx]
#         waves = np.load(file_path)
        
#         image = self.apply_qtransform(waves, self.wave_transform)
#         # print(image.shape)
#         # if self.transform:
#         #     image = image.squeeze().numpy()
#         #     image = A.Normalize()(image=image)["image"]

#         if self.transform:
#             i0 = albumentations.CoarseDropout(p=0.5, min_holes=5, max_height=12, max_width=12)(image=image[0])
#             image[0] = i0['image']
#             # i0 = image[0].squeeze()
#             # image[0] = albumentations.Normalize()(image=i0)["image"]

#             i1 = albumentations.CoarseDropout(p=0.5, min_holes=5, max_height=12, max_width=12)(image=image[1])
#             image[1] = i1['image']
#             # i1 = image[1].squeeze()
#             # image[1] = albumentations.Normalize()(image=i1)["image"]

#             i2 = albumentations.CoarseDropout(p=0.5, min_holes=5, max_height=12, max_width=12)(image=image[2])
#             image[2] = i2['image']
#             # i2 = image[2].squeeze()
#             # image[2] = albumentations.Normalize()(image=i2)["image"]

#         # image = image.squeeze()
#         # image = albumentations.Normalize()(image=image)["image"]

#         label = torch.tensor(self.labels[idx]).float()
#         # print(image.shape)
#         return image, label

class ThreeTrainDataset(Dataset):
    def __init__(self, CFG, df, transform=False):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["target"].values
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def apply_qtransform(self, waves):
        hide = random.randint(0, 5)

        w0 = waves[0]
        if self.transform:
            if hide == 0:
                w0 = np.zeros(w0.shape)

        w0 = w0 / 4.615211621383077e-20# 7.422368145063434e-21 #4.615211621383077e-20
        w0 = torch.from_numpy(w0).float()

        w1 = waves[1]
        if self.transform:
            if hide == 1:
                w1 = np.zeros(w1.shape)
        w1 = w1 / 4.1438353591025024e-20 #7.418562450079042e-21 #4.1438353591025024e-20
        w1 = torch.from_numpy(w1).float()

        w2 = waves[2]
        if self.transform:
            if hide == 2:
                w2 = np.zeros(w2.shape)
        w2 = w2 / 6e-20 #1.837612126304118e-21 #6e-20
        w2 = torch.from_numpy(w2).float()
        return w0, w1, w2

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        
        w0, w1, w2 = self.apply_qtransform(waves)
        
        label = torch.tensor(self.labels[idx]).float()
        return w0, w1, w2, label