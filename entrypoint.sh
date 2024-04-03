#!/bin/bash

cd /app
cd ./dataset/data
ls -la
cd /app
dvc pull
dvc repro -f
git config --global user.email "automatic@pepemoss.com"
git config --global user.name "Automatic Pepemoss"
git status
git add .
git commit -m "automatic"
git push https://$CI_COMMIT_USERNAME:$CI_COMMIT_PASSWORD@git.pepemoss.com:universitat/ml/sam_train.git HEAD:$CI_COMMIT_BRANCH
dvc push