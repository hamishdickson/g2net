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

warnings.filterwarnings("ignore")

class CFG:
    trial = 999
    seed = 42
    n_fold = 5
    epochs = [5 for _ in range(10)]
    batch_size = [64 for _ in range(10)]
    num_workers = 4
    model_name = "tf_efficientnet_b1_ns"
    target_size = 1
    lr = [3e-3]
    resolution = [16 for _ in range(10)]
    d0_norm = 5e-20
    d1_norm = 5e-20
    d2_norm = 6e-20
    pretrained = True
    batch_normed = False
    weight_decay = [1e-5 for _ in range(10)]
    # max_grad_norm = 1000
    es_round = 3
    input_shape = "3d"
    trials = 1
    sample = False



def train_loop(folds, fold=0):
    writer = SummaryWriter()
    if CFG.sample:
        folds = folds.sample(frac=0.01)
    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds["target"].values


    train_dataset = datasets.TrainDataset(
        CFG, train_folds, transform=False
    )
    valid_dataset = datasets.ValidDataset(
        CFG, valid_folds
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

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=CFG.lr #, weight_decay=CFG.weight_decay
    # )
    # optimizer = utils.MADGRAD(
    #     model.parameters(), lr=CFG.lr #, weight_decay=CFG.weight_decay
    # )
    # optimizer = utils.RAdam(
    #     model.parameters(), lr=CFG.lr #, weight_decay=CFG.weight_decay
    # )
    optimizer = transformers.AdamW(
        model.parameters(), lr=CFG.lr[trial], weight_decay=CFG.weight_decay[trial]
    )
    # optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)

    # optimizer = utils.Lookahead(optimizer, k=5, alpha=0.5)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0, #0.06*CFG.epochs*len(train_loader),
        num_training_steps=CFG.epochs[trial]*len(train_loader)
    )

    # scheduler = None
    
    # optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
    #                                           pct_start=0.1, 
    #                                           div_factor=1e3, 
    #                                           max_lr=1e-4, 
    #                                           epochs=CFG.epochs, 
    #                                           steps_per_epoch=len(train_loader)
    # )

    criterion = nn.BCEWithLogitsLoss()
    # criterion = utils.BCEFocalLoss()

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

    return valid_folds


def get_result(result_df):
    preds = result_df["preds"].values
    labels = result_df["target"].values
    score = utils.get_score(labels, preds)
    print(f"Score: {score:<.4f}")


def multiprocess_wrapper(folds, fold, queue):
    out_df = train_loop(folds, fold)
    queue.put(out_df)

if __name__ == "__main__":
    print("starting training")

    utils.set_seeds(CFG.seed)

    train = pd.read_csv("input/train_folds.csv")

    
    for trial in range(CFG.trials):
        CFG.trial = trial
        print(f"training trial {trial}")

        oof_df = pd.DataFrame()
        # all_oofs = []
        # q = Queue()
        # processes = []

        # for fold in [0, 1, 2, 3, 4]:
        #     p = Process(target=multiprocess_wrapper, args=(train, fold, q))
        #     processes.append(p)
        #     p.start()

        # for p in processes:
        #     ret = q.get()
        #     all_oofs.append(ret)


        for fold in [0]:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            get_result(_oof_df)

        print(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv("models/oof_df.csv", index=False)
