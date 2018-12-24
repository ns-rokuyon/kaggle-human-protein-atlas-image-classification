#!/bin/bash

cd $HOME

COMPETITION_NAME='human-protein-atlas-image-classification'

cp kaggle-$COMPETITION_NAME/*.py $HOME/
cp kaggle-$COMPETITION_NAME/*.sh $HOME/

rm workspace.py
rm workspace_colab.py
