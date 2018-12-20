import tqdm
import torch
import gc
import datetime
import torch.nn.functional as F

import infer
from model import freeze_backbone, unfreeze
from data import progress_bar, save_bestmodel, write_log, save_checkpoint
from loss import FocalLoss, f1_loss
from optim import CosineLRWithRestarts


training_log_format = '[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'


def train(model, optimizer, n_epoch, train_iter, val_iter,
          scheduler,
          device=None,
          early_stopping_limit=100,
          logging_interval=50,
          epoch_break_at=None,
          focal_alpha=1.0,
          model_keyname='model',
          criterion='bce'):

    assert isinstance(scheduler, CosineLRWithRestarts)

    best_score = 0.0
    n_stay = 0

    if criterion in ('focal', 'focal_and_f1', 'focal_and_bce'):
        focal_loss_func = FocalLoss()

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

        for batch_idx, (x, t) in enumerate(train_iter):
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()

            # Forward
            logit = model(x)

            if criterion == 'bce':
                loss = F.binary_cross_entropy_with_logits(logit, t)
            elif criterion == 'focal':
                loss = focal_loss_func(logit, t)
            elif criterion == 'f1':
                loss = f1_loss(logit, t)
            elif criterion == 'focal_and_f1':
                loss = focal_loss_func(logit, t) + f1_loss(logit, t)
            elif criterion == 'focal_and_bce':
                loss = focal_alpha * focal_loss_func(logit, t) + F.binary_cross_entropy_with_logits(logit, t)
            else:
                raise ValueError(criterion)

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