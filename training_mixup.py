import tqdm
import torch
import gc
import datetime
import torch.nn.functional as F
import numpy as np

import infer
from model import freeze_backbone, unfreeze
from data import progress_bar, save_bestmodel, write_log, save_checkpoint
from loss import naive_cross_entropy_loss, f1_loss
from optim import CosineLRWithRestarts


training_log_format = '[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'


def mixup(x1, t1, x2, t2, alpha=0.4, share_lambda=False):
    bs = x1.size()[0]

    if share_lambda:
        _lambda = np.random.beta(alpha, alpha).astype(np.float32)
    else:
        _lambda = np.random.beta(alpha, alpha, size=bs).astype(np.float32)

    x = _lambda.reshape(bs, 1, 1, 1) * x1 + (1 - _lambda.reshape(bs, 1, 1, 1)) * x2
    t = _lambda.reshape(bs, 1) * t1 + (1 - _lambda.reshape(bs, 1)) * t2

    return x, t


def train(model, optimizer, n_epoch, train_iters, val_iter,
          scheduler,
          device=None,
          early_stopping_limit=100,
          logging_interval=50,
          epoch_break_at=None,
          alpha=0.4,
          share_lambda=False,
          model_keyname='model'):

    assert isinstance(scheduler, CosineLRWithRestarts)
    assert isinstance(train_iters, list)

    best_score = 0.0
    n_stay = 0

    for epoch in range(n_epoch):
        model.train()

        gc.collect()
        torch.cuda.empty_cache()

        scheduler.step()
        print(f'scheduler.step() at start of {epoch}')

        for g in optimizer.param_groups:
            current_lr = g['lr']
            print(f'Current LR: {current_lr}')

        total_loss = 0
        total_size = 0

        for batch_idx, ((x1, t1), (x2, t2)) in enumerate(zip(*train_iters)):
            x, t = mixup(x1, t1, x2, t2, alpha=alpha, share_lambda=share_lambda)

            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()

            # Forward
            logit = model(x)

            loss = naive_cross_entropy_loss(logit, t)

            total_loss += loss.item()
            total_size += x.size(0)

            # Backward
            loss.backward()
            optimizer.step()

            scheduler.batch_step()

            if batch_idx % logging_interval == 0:
                now = datetime.datetime.now()
                print(training_log_format.format(
                    now, epoch,
                    batch_idx * len(x),
                    len(train_iter.dataset),
                    100.0 * batch_idx / len(train_iter),
                    total_loss / total_size
                ))

            if epoch_break_at and batch_idx >= epoch_break_at:
                print('Break epoch at {}'.format(batch_idx))
                optimizer.zero_grad()
                break

        gc.collect()
        torch.cuda.empty_cache()

        # Evaluation
        score = infer.evaluate(model, val_iter, device=device)
        eval_message = '[{}] Train Epoch: {}, F1: {:.6f} %'.format(now, epoch, score)
        write_log(eval_message, keyname=model_keyname)

        if best_score < score:
            best_score = score
            save_bestmodel(model, model_keyname)

            message = 'Saved model at {} (Best Score: {:.6f})'.format(epoch, best_score)
            write_log(message, keyname=model_keyname)
            n_stay = 0
        else:
            n_stay += 1

        save_checkpoint(model, model_keyname, optimizer=optimizer, scheduler=scheduler)
        write_log(f'Saved checkpoint at {epoch}', keyname=model_keyname)

        if n_stay >= early_stopping_limit:
            write_log('Early stopping at {} (Best Score: {:.6f})'.format(
                epoch, best_score
            ), keyname=model_keyname)
            break

    return model, best_score