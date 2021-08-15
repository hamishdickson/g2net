import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import GradScaler

import transformers

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddShortNoises, Gain

from . import engine
from . import datasets
from . import models
from . import utils

import warnings

warnings.filterwarnings("ignore")

# class CFG:
#     seed = 42
#     n_fold = 5
#     epochs = 4
#     batch_size = 64
#     num_workers = 32
#     model_name = "tf_efficientnetv2_l"
#     target_size = 1
#     lr = 1e-3
#     weight_decay = 1e-5
#     max_grad_norm = 1000
#     es_round = 3
#     input_shape = "3d"

class CFG:
    seed = 42
    n_fold = 5
    epochs = 6
    batch_size = 64
    num_workers = 42
    model_name = "tf_efficientnet_b7_ns"
    target_size = 1
    lr = 1e-3
    weight_decay = 1e-5
    max_grad_norm = 1000
    es_round = 3
    input_shape = "3d"

# from 2012.12877
# class CFG:
#     seed = 42
#     n_fold = 5
#     epochs = 10
#     batch_size = 64
#     num_workers = 42
#     model_name = "deit_base_distilled_patch16_224"
#     target_size = 1
#     lr = 5e-4*(batch_size)/512
#     weight_decay = 0.05
#     max_grad_norm = 1000
#     es_round = 3
#     input_shape = "3d"

def get_transforms(*, data):

    if data == "train":
        return A.Compose(
            [   
                # A.Resize(57, 384),
                ToTensorV2(),
            ]
        )
    elif data == "audio":
        return Compose([
            AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.0015, p=0.2),
            AddGaussianSNR(p=0.2),
            # Gain(min_gain_in_db=-15,max_gain_in_db=15,p=0.3)
        ])

    elif data == "valid":
        return A.Compose(
            [   
                # A.Resize(57, 384),
                # ToTensorV2(),
            ]
        )


def train_loop(folds, fold):
    writer = SummaryWriter()
    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds["target"].values

    if CFG.input_shape == "3d":
        train_dataset = datasets.ThreeTrainDataset(
            CFG, train_folds, transform=None #get_transforms(data="audio")
        )
        valid_dataset = datasets.ThreeTrainDataset(
            CFG, valid_folds, transform=None
        )
    else:
        train_dataset = datasets.TrainDataset(
            CFG, train_folds, transform=get_transforms(data="valid")
        )
        valid_dataset = datasets.TrainDataset(
            CFG, valid_folds, transform=get_transforms(data="valid")
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=42,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if "deit" in CFG.model_name:
        model = models.ViTModel(CFG, pretrained=True)
    elif "v2" in CFG.model_name:
        model = models.V2Model(CFG, pretrained=True)
    elif CFG.input_shape == "3d":
        model = models.V2Model(CFG, pretrained=True)
    else:
        model = models.CustomModel(CFG, pretrained=True)
    model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr #, weight_decay=CFG.weight_decay
    )
    # optimizer = transformers.AdamW(
    #     model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    # )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0, #0.06*CFG.epochs*len(train_loader),
        num_training_steps=4*len(train_loader)
    )

    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_loss = np.inf

    es_count = 0

    scaler = GradScaler()

    es = utils.EarlyStopping(patience=100)

    for epoch in range(CFG.epochs):
        ave_train_loss = engine.train_fn(
            epoch, 
            fold,
            CFG, 
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            scaler, 
            writer, 
            valid_loader,
            es
        )

        ave_valid_loss, preds, score = engine.valid_fn(valid_loader, model, criterion)

        writer.add_scalar('Loss/train', ave_train_loss, epoch)
        writer.add_scalar('Loss/valid', ave_valid_loss, epoch)
        writer.add_scalar('roc', score, epoch)
        writer.add_scalar('Loss/valid2', ave_valid_loss, CFG.batch_size*len(train_loader)*(epoch + 1)/48)
        writer.add_scalar('roc2', score, CFG.batch_size*len(train_loader)*(epoch + 1)/48)

        print(f"results for epoch {epoch + 1}: score {score}")

        es(score, model, f"models/{CFG.model_name}_fold{fold}_best_score.pth", preds)

        print(f"es count {es_count}")

        if es_count > CFG.es_round:
            print("early stopping")
            break

    valid_folds["preds"] = torch.load(
        f"models/{CFG.model_name}_fold{fold}_best_score.pth",
        map_location=torch.device("cpu"),
    )["preds"]

    return valid_folds


def get_result(result_df):
    preds = result_df["preds"].values
    labels = result_df["target"].values
    score = utils.get_score(labels, preds)
    print(f"Score: {score:<.4f}")


if __name__ == "__main__":
    print("starting training")

    utils.set_seeds(CFG.seed)

    train = pd.read_csv("input/train_folds.csv")

    oof_df = pd.DataFrame()
    for fold in [0]:
        print(f"training fold {fold}")
        _oof_df = train_loop(train, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        get_result(_oof_df)

    print(f"========== CV ==========")
    get_result(oof_df)
    # save result
    oof_df.to_csv("models/oof_df.csv", index=False)
