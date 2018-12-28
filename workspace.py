from pathlib import Path


# Dataset root dir
data_root_dir = Path('D:/Users/ns/.kaggle/competitions/human-protein-atlas-image-classification')

# Train set
train_image_dir = data_root_dir / 'train'
train_csv = data_root_dir / 'train.csv'

# Test set
test_image_dir = data_root_dir / 'test'
test_csv = data_root_dir / 'sample_submission.csv'

# Workspace
workspace_dir = Path('D:/Users/ns/git_repos/kaggle-human-protein-atlas-image-classification')

# Directory to save models
model_dir = workspace_dir / 'models'

# Directory to save pre-splited set
list_dir = workspace_dir / 'lists'

# Directory to submission file
submission_dir = workspace_dir / 'submissions'

# Directory to save log file
log_dir = workspace_dir / 'logs'

# kfold list dir
kfold_list_dir = list_dir / 'kfold'
kfold_cv5_list_dir = kfold_list_dir / 'cv5'
multilabel_stratified_kfold_cv3_list_dir = kfold_list_dir / 'mls_cv3'
multilabel_stratified_kfold_cv5_list_v2_dir = kfold_list_dir / 'mls_cv5_v2'

# Training-Validation set
#holdout_train_csv = list_dir / 'holdout_train.csv'
#holdout_val_csv = list_dir / 'holdout_val.csv'

# hdf5 file
images_h5_file = list_dir / 'images.h5'
test_images_h5_file = list_dir / 'test_images.h5'

# Ex data
ex_image_dir = data_root_dir / 'ex'
ex_csv = list_dir / 'HPAv18RBGY_wodpl.csv'
ex_images_h5_file = list_dir / 'ex_images.h5'

# Enhanced dataset
enhanced_train_csv = list_dir / 'enhanced_train.csv'
undersampled_enhanced_train_csv = list_dir / 'undersampled_enhanced_train.csv'
mls_undersampled_enhanced_kfold_cv5_list_dir = kfold_list_dir / 'mls_us_enh_cv5_v2'

# Enhanced full dataset
enhanced_full_train_csv = list_dir / 'enhanced_full_train.csv'
ex_full_images_h5_file = list_dir / 'ex_full_images.h5'


def setup():
    if not workspace_dir.exists():
        raise RuntimeError('Workspace dir is not found: {}'.format(workspace_dir))

    model_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    kfold_cv5_list_dir.mkdir(parents=True, exist_ok=True)
    multilabel_stratified_kfold_cv3_list_dir.mkdir(parents=True, exist_ok=True)
    multilabel_stratified_kfold_cv5_list_v2_dir.mkdir(parents=True, exist_ok=True)
    ex_image_dir.mkdir(parents=True, exist_ok=True)
    mls_undersampled_enhanced_kfold_cv5_list_dir.mkdir(parents=True, exist_ok=True)