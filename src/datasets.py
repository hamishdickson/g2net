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
#     def __init__(self, CFG, df, transform=False, image_transform=None):
#         self.df = df
#         self.file_names = df["file_path"].values
#         self.labels = df["target"].values
#         # TRYME worth playing with these values
#         self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=512, hop_length=16)
#         self.transform = transform

#         self.image_transform = image_transform
        
#         self.no_waves = df[df['target'] == 0].reset_index(drop=True)
#         self.no_waves_file_names = self.no_waves["file_path"].values

#     def __len__(self):
#         return len(self.df)

#     def apply_qtransform(self, waves, qtransform):
#         hide = random.randint(0, 5)
#         reverser = random.randint(0, 1)
        

#         w0 = waves[0]
#         if self.transform:
#             if hide == 0:
#                 random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
#                 random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
#                 w0 = random_no_wave[0]
#                 # w0 = np.zeros(w0.shape)
                
#         w0 = w0 / 4.615211621383077e-20# 7.422368145063434e-21 #4.615211621383077e-20
#         w0 = torch.from_numpy(w0).float()
#         i0 = qtransform(w0)

#         w1 = waves[1]
#         if self.transform:
#             if hide == 1:
#                 random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
#                 random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
#                 w1 = random_no_wave[1]
#                 # w1 = np.zeros(w1.shape)
#         w1 = w1 / 4.1438353591025024e-20 #7.418562450079042e-21 #4.1438353591025024e-20
#         w1 = torch.from_numpy(w1).float()
#         i1 = qtransform(w1)

#         w2 = waves[2]
#         if self.transform:
#             if hide == 2:
#                 random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
#                 random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
#                 w2 = random_no_wave[2]
#                 # w2 = np.zeros(w2.shape)
#         w2 = w2 / 6e-20 #1.837612126304118e-21 #6e-20
#         w2 = torch.from_numpy(w2).float()
#         i2 = qtransform(w2)

#         image = np.vstack([i0, i1, i2])

#         return image


#     def __getitem__(self, idx):
#         file_path = self.file_names[idx]
#         waves = np.load(file_path)
        
#         image = self.apply_qtransform(waves, self.wave_transform)

#         # if self.transform:
#         #     i0 = albumentations.CoarseDropout(p=0.5, min_holes=5, max_height=12, max_width=12)(image=image[0])
#         #     image[0] = i0['image']

#         #     i1 = albumentations.CoarseDropout(p=0.5, min_holes=5, max_height=12, max_width=12)(image=image[1])
#         #     image[1] = i1['image']

#         #     i2 = albumentations.CoarseDropout(p=0.5, min_holes=5, max_height=12, max_width=12)(image=image[2])
#         #     image[2] = i2['image']
        
#         # image = image.squeeze()
#         # print(image.shape)
#         # image = albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image=image)["image"]


#         image = self.image_transform(image=image.T)['image']
#         label = torch.tensor(self.labels[idx]).float()
#         return image, label

class ThreeTrainDataset(Dataset):
    def __init__(self, CFG, df, transform=False):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["target"].values
        self.transform = transform

        self.no_waves = df[df['target'] == 0].reset_index(drop=True)
        self.no_waves_file_names = self.no_waves["file_path"].values

        self.X_0_mean = np.load("input/X_0_mean.npy")
        self.X_1_mean = np.load("input/X_1_mean.npy")
        self.X_2_mean = np.load("input/X_2_mean.npy")


    def __len__(self):
        return len(self.df)

    def apply_qtransform(self, waves):
        hide = random.randint(0, 14)
        reverser = random.randint(0, 1)
        scale = 1

        w0 = waves[0]
        if self.transform:
            # if reverser == 1:
            #     w0 = np.flip(w0)
            if hide == 0:
                random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
                random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
                w0 = random_no_wave[0]
                # w0 = np.zeros(w0.shape)
                
        w0 = scale * w0 / 5e-20 #3e-21 #4.615211621383077e-20# 7.422368145063434e-21
        # w0 = scale * w0 / (self.X_0_mean / 1e-6)
        w0 = torch.from_numpy(w0).float()

        w1 = waves[1]
        if self.transform:
            # if reverser == 1:
            #     w1 = np.flip(w1)
            if hide == 1:
                random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
                random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
                w1 = random_no_wave[1]
                # w1 = np.zeros(w1.shape)
        w1 = scale * w1 / 5e-20 #2e-21 # 4.1438353591025024e-20 #7.418562450079042e-21

        w1 = torch.from_numpy(w1).float()

        w2 = waves[2]
        if self.transform:
            # if reverser == 1:
            #     w2 = np.flip(w2)
            if hide == 2:
                random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
                random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
                w2 = random_no_wave[2]
                # w2 = np.zeros(w2.shape)
        w2 = scale * w2 / 6e-20 #3.5e-21 #6e-20 #1.837612126304118e-21

        w2 = torch.from_numpy(w2).float()
        return w0, w1, w2

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        
        w0, w1, w2 = self.apply_qtransform(waves)
        
        label = torch.tensor(self.labels[idx]).float()
        return w0, w1, w2, label