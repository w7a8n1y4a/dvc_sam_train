#!/bin/bash

cp -r /app/sam_train /
cd /sam_train

ls -la

git status
git fetch --all
git pull

git checkout $CI_COMMIT_BRANCH

dvc repro -f

git config --global user.email "automatic@pepemoss.com"
git config --global user.name "Automatic Pepemoss"
git add dvc.lock
git commit -m "automatic"

git push https://$CI_COMMIT_USERNAME:$CI_COMMIT_PASSWORD@git.pepemoss.com/universitat/ml/sam_train.git

dvc push