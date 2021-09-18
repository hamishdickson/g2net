import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from nnAudio.Spectrogram import CQT1992v2
import albumentations
import random
import scipy

from torch_audiomentations import AddColoredNoise

import colorednoise as cn

class TrainDataset(Dataset):
    def __init__(self, CFG, df, transform=False):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["target"].values
        self.transform = transform
        self.cfg = CFG

        self.no_waves = df[df['target'] == 0].reset_index(drop=True)
        self.no_waves_file_names = self.no_waves["file_path"].values

        self.bHp, self.aHp = scipy.signal.butter(8, (20, 512), btype="bandpass", fs=2048)

        self.noise = AddColoredNoise(min_f_decay=0.8, max_f_decay=1.2, p=0.4)

    def __len__(self):
        return len(self.df)

    def apply_qtransform(self, waves):
        hide = random.randint(0, 5)

        w0 = waves[0]
        if self.transform:
            if hide == 0:
                # random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
                # random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
                # w0 = random_no_wave[0]
                w0 = np.zeros(w0.shape)

        w0 = w0 / self.cfg.d0_norm
        w0 = scipy.signal.lfilter(self.bHp, self.aHp, w0)
        w0 = w0 * scipy.signal.tukey(4096, alpha=0.25)

        w0 = torch.from_numpy(w0).float()
        if self.transform:
            w0 = self.noise(w0.unsqueeze(0).unsqueeze(1), sample_rate=2048).squeeze(0).squeeze(0)

        w1 = waves[1]
        if self.transform:
            if hide == 1:
                # random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
                # random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
                # w1 = random_no_wave[1]
                w1 = np.zeros(w1.shape)
        w1 = w1 /self.cfg.d1_norm
        w1 = scipy.signal.lfilter(self.bHp, self.aHp, w1) 

        w1 = w1 * scipy.signal.tukey(4096, alpha=0.25)
        w1 = torch.from_numpy(w1).float()
        if self.transform:
            w1 = self.noise(w1.unsqueeze(0).unsqueeze(1), sample_rate=2048).squeeze(0).squeeze(0)


        w2 = waves[2]
        if self.transform:
            if hide == 2:
                # random_no_wave_idx = random.randint(0, len(self.no_waves) - 1)
                # random_no_wave = np.load(self.no_waves_file_names[random_no_wave_idx])
                # w2 = random_no_wave[2]
                w2 = np.zeros(w2.shape)
        w2 = w2 / self.cfg.d2_norm
        w2 = scipy.signal.lfilter(self.bHp, self.aHp, w2)

        w2 = w2 * scipy.signal.tukey(4096, alpha=0.25)
        w2 = torch.from_numpy(w2).float()

        if self.transform:
            w2 = self.noise(w2.unsqueeze(0).unsqueeze(1), sample_rate=2048).squeeze(0).squeeze(0)

        return w0, w1, w2

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        
        w0, w1, w2 = self.apply_qtransform(waves)
        
        label = torch.tensor(self.labels[idx]).float()
        return w0, w1, w2, label
