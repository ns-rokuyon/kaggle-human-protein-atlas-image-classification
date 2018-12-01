import tqdm
import torch
import gc
import datetime
import torch.nn.functional as F

import infer
from data import progress_bar, save_model
from loss import FocalLoss, f1_loss


training_log_format = '[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'


def train(model, optimizer, n_epoch, train_iter, val_iter,
          device=None,
          early_stopping_limit=100,
          reduce_limit=10,
          reduce_step=0.5,
          base_lr=0.01,
          min_lr=0.000001,
          logging_interval=50,
          epoch_break_at=None,
          class_weights=None,
          model_keyname='model',
          criterion='bce'):
    best_score = 0.0
    n_stay = 0
    current_lr = base_lr

    if criterion in ('focal', 'focal_and_f1'):
        focal_loss_func = FocalLoss()

    for epoch in range(n_epoch):
        model.train()

        gc.collect()
        torch.cuda.empty_cache()

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
                loss = loss_func(logit, t)
            elif criterion == 'f1':
                loss = f1_loss(logit, t)
            elif criterion == 'focal_and_f1':
                loss = focal_loss_func(logit, t) + f1_loss(logit, t)
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
        print('[{}] Train Epoch: {}, F1: {:.6f} %'.format(
            now, epoch, score
        ))

        if best_score < score:
            best_score = score
            save_model(model, model_keyname)
            print('Saved model at {} (Best Score: {:.6f})'.format(
                epoch, best_score
            ))
            n_stay = 0
        else:
            n_stay += 1

        if n_stay >= early_stopping_limit:
            print('Early stopping at {} (Best Score: {:.6f})'.format(
                epoch, best_score
            ))
            break

        if current_lr > min_lr and n_stay >= reduce_limit:
            current_lr = reduce_step * current_lr
            for g in optimizer.param_groups:
                g['lr'] = current_lr
            n_stay = 0
            print('Reduce lr at {} (to: {})'.format(epoch, current_lr))

    return model, best_score, current_lr