import numpy as np
import torch
from tqdm import tqdm
from . import utils

from torch.cuda.amp import autocast

def train_fn(epoch, CFG, model, train_loader, criterion, optimizer, scheduler, scaler, writer):
    losses = utils.AverageMeter()
    # switch to train model
    model.train()

    tk0 = tqdm(train_loader, total=len(train_loader))

    for idx, (images, labels) in enumerate(tk0):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with autocast():
            y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)

        losses.update(loss.item(), labels.size(0))

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        if scheduler:
            scheduler.step()
        scaler.update()
        
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        tk0.set_postfix(train_loss=losses.avg)

        if idx % 100 == 0:
            writer.add_scalar(f'Loss/mid-train_{epoch}', losses.avg, idx*CFG.batch_size/48)

    return losses.avg


def valid_fn(valid_loader, model, criterion):
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []

    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for images, labels in tk0:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        batch_size = labels.size(0)
        # compute loss
        with autocast():
            with torch.no_grad():
                y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to("cpu").numpy())

        tk0.set_postfix(valid_loss=losses.avg)

    predictions = np.concatenate(preds)
    return losses.avg, predictions
