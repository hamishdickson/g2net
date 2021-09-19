import numpy as np
import pandas as pd

from multiprocessing import Process
from multiprocessing import Queue

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import GradScaler
import adabound
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

import optuna

warnings.filterwarnings("ignore")

class CFG:
    trial = 999
    seed = 42
    n_fold = 1
    epochs = [5 for _ in range(10)]
    batch_size = [64 for _ in range(10)]
    num_workers = 4
    model_name = "tf_efficientnet_b2_ns"
    target_size = 1
    lr = [3e-3 for _ in range(10)]
    resolution = [16 for _ in range(10)]
    d0_norm = 4.5e-20
    d1_norm = 7.12e-20
    d2_norm = 3.78e-20
    pretrained = True
    batch_normed = False
    weight_decay = [1e-8 for _ in range(10)]
    max_grad_norm = 100
    es_round = 3
    input_shape = "3d"
    trials = 1
    sample = False



def train_loop(folds, fold=0):
    writer = SummaryWriter()
    if CFG.sample:
        folds = folds.sample(frac=0.25)
    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds["target"].values


    train_dataset = datasets.TrainDataset(
        CFG, train_folds, transform=True
    )
    valid_dataset = datasets.TrainDataset(
        CFG, valid_folds, transform=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size[CFG.trial],
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size[CFG.trial] * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    if ("swin" in CFG.model_name) or ("deit" in CFG.model_name):
        model = models.ViTModel(CFG, pretrained=CFG.pretrained)
    else:
        model = models.V2Model(CFG, pretrained=CFG.pretrained)
    model.cuda()

    optimizer = transformers.AdamW(
        model.parameters(), lr=CFG.lr[trial], weight_decay=CFG.weight_decay[trial]
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=CFG.lr[trial], weight_decay=CFG.weight_decay[trial], momentum=0.9
    # )

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0, #0.06*CFG.epochs*len(train_loader),
        num_training_steps=CFG.epochs[trial]*len(train_loader)
    )

    # scheduler = None

    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_loss = np.inf

    es_count = 0

    scaler = GradScaler()

    es = utils.EarlyStopping(patience=100)

    for epoch in range(CFG.epochs[trial]):
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
            es,
            utils.AutoClip(0.0015)
        )

        ave_valid_loss, preds, score = engine.valid_fn(valid_loader, model, criterion)

        writer.add_scalar('Loss/train', ave_train_loss, epoch)
        writer.add_scalar('Loss/valid', ave_valid_loss, epoch)
        writer.add_scalar('roc', score, epoch)
        writer.add_scalar('Loss/valid2', ave_valid_loss, CFG.batch_size[CFG.trial]*len(train_loader)*(epoch + 1)/48)
        writer.add_scalar('roc2', score, CFG.batch_size[CFG.trial]*len(train_loader)*(epoch + 1)/48)

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

    return es.best_score, valid_folds


def get_result(result_df):
    preds = result_df["preds"].values
    labels = result_df["target"].values
    score = utils.get_score(labels, preds)
    print(f"Score: {score:<.4f}")


def objective(trial):
    params = {
        "d0": trial.suggest_discrete_uniform("d0", 1e-21, 1e-19, 1e-22),
        "d1": trial.suggest_discrete_uniform("d1", 1e-21, 1e-19, 1e-22),
        "d2": trial.suggest_discrete_uniform("d2", 1e-21, 1e-19, 1e-22),
    }

    CFG.d0_norm = params['d0']
    CFG.d1_norm = params['d1']
    CFG.d2_norm = params['d2']

    print(params)

    scores = []

    for fold in range(1):
        fold_score, _ = train_loop(train, fold)
        scores.append(fold_score)
            
    print(scores)
    print(f"CV score: {np.array(scores).mean()}")
    return np.array(scores).mean()

if __name__ == "__main__":
    print("starting training")

    utils.set_seeds(CFG.seed)

    train = pd.read_csv("input/train_folds.csv")

    
    for trial in range(CFG.trials):
        CFG.trial = trial
        print(f"training trial {trial}")

        # study = optuna.create_study(direction="maximize")
        # study.optimize(objective, n_trials=25)
        # print("best params ", study.best_params)
        
        oof_df = pd.DataFrame()

        for fold in range(CFG.n_fold):
            _, _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            get_result(_oof_df)

        print(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv("models/oof_df.csv", index=False)
