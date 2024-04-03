#!/bin/bash

cd /app
dvc pull
dvc repro -f
git add .
git commit -m "automatic"