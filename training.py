import tqdm
import torch
import gc
import datetime
import torch.nn.functional as F

import infer
from model import freeze_backbone, unfreeze
from data import progress_bar, save_model, write_log
from loss import FocalLoss, f1_loss


training_log_format = '[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'


def train(model, optimizer, n_epoch, train_iter, val_iter,
          device=None,
          early_stopping_limit=100,
          reduce_limit=10,
          reduce_step=0.5,
          min_lr=0.000001,
          logging_interval=50,
          epoch_break_at=None,
          class_weights=None,
          scheduler=None,
          freeze_epoch=0,
          focal_alpha=1.0,
          model_keyname='model',
          criterion='bce'):
    best_score = 0.0
    n_stay = 0

    for g in optimizer.param_groups:
        base_lr = g['lr']
        print(f'Base LR: {base_lr}')

    current_lr = base_lr

    if criterion in ('focal', 'focal_and_f1', 'focal_and_bce'):
        focal_loss_func = FocalLoss()

    for epoch in range(n_epoch):
        model.train()

        gc.collect()
        torch.cuda.empty_cache()

        if freeze_epoch > 0:
            if epoch == 0:
                freeze_backbone(model)
                write_log(f'Freeze backbone at {epoch}', keyname=model_keyname)
            elif epoch == freeze_epoch:
                unfreeze(model)
                write_log(f'Unfreeze model at {epoch}', keyname=model_keyname)

        total_loss = 0
        total_size = 0

        for batch_idx, (x, t) in enumerate(train_iter):
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()

            # Forward
            logit = model(x)

            if criterion == 'bce':
                loss = F.binary_cross_entropy_with_logits(logit, t)
            elif criterion == 'bce_weighten':
                pos_weight = torch.Tensor(class_weights)
                pos_weight = pos_weight.to(device)
                loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=pos_weight)
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
            save_model(model, model_keyname, optimizer=optimizer, scheduler=scheduler)

            message = 'Saved model at {} (Best Score: {:.6f})'.format(epoch, best_score)
            write_log(message, keyname=model_keyname)
            n_stay = 0
        else:
            n_stay += 1

        if n_stay >= early_stopping_limit:
            write_log('Early stopping at {} (Best Score: {:.6f})'.format(
                epoch, best_score
            ), keyname=model_keyname)
            break

        if scheduler:
            scheduler.step(score)

            new_lr = 0
            for g in optimizer.param_groups:
                new_lr = g['lr']
            write_log(f'Scheduler new lr: {new_lr}',
                      keyname=model_keyname)
        else:
            if current_lr > min_lr and n_stay >= reduce_limit:
                current_lr = reduce_step * current_lr
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                n_stay = 0
                write_log('Reduce lr at {} (to: {})'.format(epoch, current_lr), keyname=model_keyname)

    return model, best_score, current_lr