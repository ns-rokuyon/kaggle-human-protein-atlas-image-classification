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

# Training-Validation set
#holdout_train_csv = list_dir / 'holdout_train.csv'
#holdout_val_csv = list_dir / 'holdout_val.csv'

# hdf5 file
#images_h5_file = list_dir / 'images.h5'


def setup():
    if not workspace_dir.exists():
        raise RuntimeError('Workspace dir is not found: {}'.format(workspace_dir))

    model_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)
