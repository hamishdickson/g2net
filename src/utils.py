import os
import random
import numpy as np
import torch

from sklearn.metrics import roc_auc_score


def set_seeds(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=2, mode="max", delta=0.0002):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_preds = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path, epoch_preds, intra_epoch=False):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.best_preds = epoch_preds
            self.save_checkpoint(epoch_score, model, self.best_preds, model_path)
            
        elif (score < self.best_score + self.delta):
            if (not intra_epoch):
                self.counter += 1
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print("Score didn't improve")
        else:
            self.best_score = score
            self.best_preds = epoch_preds
            self.save_checkpoint(epoch_score, model, self.best_preds, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, preds, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                model_path,
            )
        self.val_score = epoch_score