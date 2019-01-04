import torch
import numpy as np
import gc
from argparse import ArgumentParser
from PIL import Image

import infer
import model as M
import training
import training_v2
from data import *
from dataset import *
from optim import AdamW
from data import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gen-h5', action='store_true',
                        help='Generate images.h5')
    parser.add_argument('--train', action='store_true',
                        help='Train')
    parser.add_argument('--cv', type=int, default=0, help='KFold index')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--model-keyname', default='test__')
    parser.add_argument('--epoch-break-at', type=int, default=0,
                        help='Break epoch at N iteration for debug')
    return parser.parse_args()


def train(model_keyname, cv, batch_size=24, epoch_break_at=None):
    print(f'Train start: model_keyname={model_keyname}, cv={cv}, '
          f'batch_size={batch_size}, epoch_break_at={epoch_break_at}')

    setup()

    torch.cuda.is_available()
    device = torch.device('cuda')
    print(f'device: {device}')

    train_df, val_df = get_mls_enhanced_full_train_val_df_fold(cv)
    train_df = oversample_brian_method(train_df)

    train_dataset = HPAEnhancedDatasetMP(train_df, size=(512, 512), use_cutout=True, cutout_ratio=0.2, use_augmentation=True)
    val_dataset = HPAEnhancedDatasetMP(val_df, size=(512, 512), use_augmentation=False)

    model = M.ResNet18v4()
    #weight = torch.load(str(model_dir / 'resnet18v3_mls_enh_full_oversampling_cosanl_rp7_bce_bs32_cutout_size512_cv0_dict.model'))
    #model.load_state_dict(weight)
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24, shuffle=False, pin_memory=True, num_workers=8)

    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-04, amsgrad=True)
    #optim_snap = torch.load(str(model_dir / 'resnet18v3_mls_enh_full_oversampling_cosanl_rp7_bce_bs32_cutout_size512_cv0_dict.optim'))
    #optimizer.load_state_dict(optim_snap)

    # Train fc only
    M.freeze_backbone(model)
    model, best_score, current_lr = training.train(model, optimizer, 1, train_loader, val_loader,
                                                   device=device,
                                                   epoch_break_at=epoch_break_at,
                                                   model_keyname=model_keyname,
                                                   criterion='bce')
    
    # Train full model
    M.unfreeze(model)
    scheduler = create_cosine_annealing_lr_scheduler(optimizer, batch_size, epoch_size=len(train_dataset),
                                                     restart_period=7, t_mult=1.0)
    #scheduler_snap = torch.load(str(model_dir / 'resnet18v3_mls_enh_full_oversampling_cosanl_rp7_bce_bs32_cutout_size512_cv0_dict.scheduler'))
    #scheduler.load_state_dict(scheduler_snap)
    model, best_score = training_v2.train(model, optimizer, 30, train_loader, val_loader, scheduler,
                                          device=device,
                                          model_keyname=model_keyname,
                                          epoch_break_at=epoch_break_at,
                                          criterion='bce')
    
    print(f'Done: bestscore={best_score}')


def main():
    args = parse_args()

    if args.gen_h5:
        df = get_train_df()
        gen_images_h5_file(df)
        return

    if args.train:
        epoch_break_at = args.epoch_break_at if args.epoch_break_at > 0 else None
        train(model_keyname=args.model_keyname,
              cv=args.cv,
              batch_size=args.batch_size,
              epoch_break_at=epoch_break_at)


if __name__ == '__main__':
    main()
