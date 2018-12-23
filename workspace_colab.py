from pathlib import Path

COMPETITION_NAME = 'human-protein-atlas-image-classification'


# Dataset root dir
data_root_dir = Path('/content')

# Train set
train_image_dir = data_root_dir / 'train'
train_csv = data_root_dir / 'train.csv'

# Test set
test_image_dir = data_root_dir / 'test'
test_csv = data_root_dir / 'sample_submission.csv'

# Google drive dir
drive_base_dir = Path('/content/drive/My Drive')
drive_kaggle_dir = drive_base_dir / 'dev/kaggle'
project_dir = drive_kaggle_dir / COMPETITION_NAME

# Workspace
workspace_dir = Path('/content/kaggle-human-protein-atlas-image-classification')

# Directory to save models
model_dir = project_dir / 'models'

# Directory to save pre-splited set
list_dir = workspace_dir / 'lists'

# Directory to submission file
submission_dir = project_dir / 'submissions'

# Directory to save log file
log_dir = project_dir / 'logs'

# kfold list dir
kfold_list_dir = list_dir / 'kfold'
kfold_cv5_list_dir = kfold_list_dir / 'cv5'
multilabel_stratified_kfold_cv3_list_dir = kfold_list_dir / 'mls_cv3'
multilabel_stratified_kfold_cv5_list_v2_dir = kfold_list_dir / 'mls_cv5_v2'

# Training-Validation set
#holdout_train_csv = list_dir / 'holdout_train.csv'
#holdout_val_csv = list_dir / 'holdout_val.csv'

# hdf5 file
images_h5_file = data_root_dir / 'images.h5'

# Ex data
ex_image_dir = data_root_dir / 'ex'     # Unavailable
ex_csv = list_dir / 'HPAv18RBGY_wodpl.csv' # Unavailable
ex_images_h5_file = data_root_dir / 'ex_images.h5'

# Enhanced dataset
enhanced_train_csv = list_dir / 'enhanced_train.csv'    # Unavailable
undersampled_enhanced_train_csv = list_dir / 'undersampled_enhanced_train.csv'  # Unavailable
mls_undersampled_enhanced_kfold_cv5_list_dir = kfold_list_dir / 'mls_us_enh_cv5_v2'


def setup():
    model_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)