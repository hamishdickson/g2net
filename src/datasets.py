import numpy as np

import torch
from torch.utils.data import Dataset
from nnAudio.Spectrogram import CQT1992v2


class TrainDataset(Dataset):
    def __init__(self, CFG, df, transform=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["target"].values
        # TRYME worth playing with these values
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=16)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def apply_qtransform(self, waves, transform):
        waves = np.hstack(waves)
        # TRYME maybe worth playing with this scaling
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
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



class ThreeTrainDataset(Dataset):
    def __init__(self, CFG, df, transform=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["target"].values
        # TRYME worth playing with these values
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def apply_qtransform(self, waves, transform):
        w0 = waves[0]
        w0 = w0 / 4.615211621383077e-20 #np.max(w0)
        w0 = torch.from_numpy(w0).float()
        i0 = transform(w0)

        w1 = waves[1]
        w1 = w1 / 4.1438353591025024e-20 #np.max(w1)
        w1 = torch.from_numpy(w1).float()
        i1 = transform(w1)

        w2 = waves[2]
        w2 = w2 / 1.1161063663761836e-20 #np.max(w2)
        w2 = torch.from_numpy(w2).float()
        i2 = transform(w2)

        image = np.vstack([i0, i1, i2])
        return image


    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        image = self.apply_qtransform(waves, self.wave_transform)
        # print(image.shape)
        # if self.transform:
        #     image = image.squeeze()
        #     image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx]).float()
        # print(image.shape)
        return image, label