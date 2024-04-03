#!/bin/bash

cd /app
cd ./dataset/data
ls -la
cd /app
dvc pull
dvc repro -f
git add .
git commit -m "automatic"