#!/bin/bash

set -e

cd $HOME

COMPETITION_NAME='human-protein-atlas-image-classification'
DRIVE_BASE_DIR=drive
DRIVE_KAGGLE_DIR=$DRIVE_BASE_DIR/dev/kaggle

pip install kaggle
mkdir $HOME/.kaggle

cp $DRIVE_KAGGLE_DIR/kaggle.json $HOME/.kaggle/

kaggle competitions list
kaggle competitions download -c $COMPETITION_NAME

unzip -q -d train -n train.zip
unzip -q -d test -n test.zip

pip install scikit-learn tqdm pandas albumentations h5py scikit-multilearn iterative-stratification Pillow
pip install pretrainedmodels==0.7.4

cp kaggle-$COMPETITION_NAME/*.py $HOME/
cp kaggle-$COMPETITION_NAME/*.sh $HOME/

rm workspace.py
rm workspace_colab.py