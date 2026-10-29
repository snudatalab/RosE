#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/fig6
: > results/fig6/results.txt

python src/baselines.py --method sasrec --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --save-recommendations results/fig6/chess_sasrec.pkl >> results/fig6/results.txt
python src/analysis.py --analysis tail-frequency --recommendations results/fig6/chess_sasrec.pkl >> results/fig6/results.txt
python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior train --save-recommendations results/fig6/chess_rose.pkl >> results/fig6/results.txt
python src/analysis.py --analysis tail-frequency --recommendations results/fig6/chess_rose.pkl >> results/fig6/results.txt
