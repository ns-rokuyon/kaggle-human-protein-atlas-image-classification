#!/bin/bash

cd $HOME

cp kaggle-$COMPETITION_NAME/*.py $HOME/
cp kaggle-$COMPETITION_NAME/*.sh $HOME/

rm workspace.py
rm workspace_colab.py