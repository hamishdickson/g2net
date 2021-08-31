import numpy as np
import torch
from tqdm import tqdm
from . import utils

from torch.cuda.amp import autocast

from nnAudio.Spectrogram import CQT1992v2

def train_fn(epoch, fold, CFG, model, train_loader, criterion, optimizer, scheduler, scaler, writer, valid_loader, es, clipper):
    losses = utils.AverageMeter()
    # switch to train model
    model.train()

    tk0 = tqdm(train_loader, total=len(train_loader))

    is_set = False

    for idx, (w0, w1, w2, labels) in enumerate(tk0):
        w0 = w0.cuda()
        w1 = w1.cuda()
        w2 = w2.cuda()
        labels = labels.cuda()

        with autocast():
            y_preds = model(w0, w1, w2)
            loss = criterion(y_preds.view(-1), labels)

        losses.update(loss.item(), labels.size(0))

        scaler.scale(loss).backward()

        clipper(model)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        
        scaler.update()
        
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        if scheduler:
            scheduler.step()

        tk0.set_postfix(train_loss=losses.avg)

        if idx % 100 == 0:
            writer.add_scalar(f'Loss/mid-train_{epoch}', losses.avg, idx*CFG.batch_size/48)

        if (epoch >= 3) and (idx % 2000 == 0) and (idx > 0):
            ave_valid_loss, preds, score = valid_fn(valid_loader, model, criterion)
            model.train()
            writer.add_scalar('Loss/valid2', ave_valid_loss, (idx+epoch*len(train_loader))*CFG.batch_size/48)
            writer.add_scalar('roc2', score, (idx+epoch*len(train_loader))*CFG.batch_size/48)

            es(score, model, f"models/{CFG.model_name}_fold{fold}_best_score.pth", preds)
            

    return losses.avg


def valid_fn(valid_loader, model, criterion):
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    _labels = []

    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for idx, (w0, w1, w2, labels) in enumerate(tk0):
        w0 = w0.cuda()
        w1 = w1.cuda()
        w2 = w2.cuda()
        labels = labels.cuda()
        batch_size = labels.size(0)
        # compute loss
        with autocast():
            with torch.no_grad():
                y_preds = model(w0, w1, w2)
            loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        _labels.append(labels.to("cpu").numpy())

        tk0.set_postfix(valid_loss=losses.avg)

    predictions = np.concatenate(preds)
    valid_labels = np.concatenate(_labels)

    score = utils.get_score(valid_labels, predictions)
    return losses.avg, predictions, score
