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
#     batch_size = 32
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
    epochs = 4
    batch_size = 32
    num_workers = 32
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
#     epochs = 2
#     batch_size = 64
#     num_workers = 32
#     model_name = "deit_base_distilled_patch16_224"
#     target_size = 1
#     lr = 1e-3 #5e-4*(batch_size)/512
#     weight_decay = 0 #0.05
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

    elif data == "valid":
        return A.Compose(
            [   
                # A.Resize(57, 384),
                ToTensorV2(),
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
            CFG, train_folds, transform=get_transforms(data="train")
        )
        valid_dataset = datasets.ThreeTrainDataset(
            CFG, valid_folds, transform=get_transforms(data="train")
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
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if "deit" in CFG.model_name:
        model = models.ViTModel(CFG, pretrained=True)
    elif "v2" in CFG.model_name:
        model = models.V2Model(CFG, pretrained=True)
    else:
        model = models.V2Model(CFG, pretrained=True)
    model.cuda()

    optimizer = transformers.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06*CFG.epochs*len(train_loader),
        num_training_steps=CFG.epochs*len(train_loader)
    )

    # scheduler = None

    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_loss = np.inf

    es_count = 0

    scaler = GradScaler()

    for epoch in range(CFG.epochs):
        ave_train_loss = engine.train_fn(epoch, CFG, model, train_loader, criterion, optimizer, scheduler, scaler, writer)

        ave_valid_loss, preds = engine.valid_fn(valid_loader, model, criterion)

        score = utils.get_score(valid_labels, preds)

        writer.add_scalar('Loss/train', ave_train_loss, epoch + 0)
        writer.add_scalar('Loss/valid', ave_valid_loss, epoch + 0)
        writer.add_scalar('roc', score, epoch + 0)

        print(f"results for epoch {epoch + 1}: score {score}")

        if score > best_score:
            best_score = score
            print(f"epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                f"models/{CFG.model_name}_fold{fold}_best_score.pth",
            )
            es_count = 0
        else:
            es_count += 1

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
    for fold in [1,2,3,4]:
        _oof_df = train_loop(train, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        get_result(_oof_df)

    print(f"========== CV ==========")
    get_result(oof_df)
    # save result
    oof_df.to_csv("models/oof_df.csv", index=False)
