import numpy as np
import torch
from tqdm import tqdm
from . import utils


def train_fn(CFG, model, train_loader, criterion, optimizer, scheduler):
    losses = utils.AverageMeter()
    # switch to train model
    model.train()

    tk0 = tqdm(train_loader, total=len(train_loader))

    for images, labels in tk0:
        images = images.cuda()
        labels = labels.cuda()

        y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)

        losses.update(loss.item(), labels.size(0))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        optimizer.step()
        scheduler.step()
        
        optimizer.zero_grad()

        tk0.set_postfix(train_loss=losses.avg, grad_norm=grad_norm)

    return losses.avg


def valid_fn(valid_loader, model, criterion):
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []

    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for images, labels in tk0:
        images = images.cuda()
        labels = labels.cuda()
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to("cpu").numpy())

        tk0.set_postfix(train_loss=losses.avg)

    predictions = np.concatenate(preds)
    return losses.avg, predictions
