#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/fig8
: > results/fig8/results.txt

python src/baselines.py --method sasrec --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --save-recommendations results/fig8/chess_sasrec.pkl >> results/fig8/results.txt
python src/analysis.py --analysis segment-ndcg --recommendations results/fig8/chess_sasrec.pkl >> results/fig8/results.txt
python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior train --save-recommendations results/fig8/chess_rose.pkl >> results/fig8/results.txt
python src/analysis.py --analysis segment-ndcg --recommendations results/fig8/chess_rose.pkl >> results/fig8/results.txt
python src/baselines.py --method sasrec --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --save-recommendations results/fig8/crypto_sasrec.pkl >> results/fig8/results.txt
python src/analysis.py --analysis segment-ndcg --recommendations results/fig8/crypto_sasrec.pkl >> results/fig8/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior train --save-recommendations results/fig8/crypto_rose.pkl >> results/fig8/results.txt
python src/analysis.py --analysis segment-ndcg --recommendations results/fig8/crypto_rose.pkl >> results/fig8/results.txt
