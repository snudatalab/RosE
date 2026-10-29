#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/fig7
: > results/fig7/results.txt

python src/baselines.py --method sasrec --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --save-recommendations results/fig7/chess_sasrec.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/chess_sasrec.pkl >> results/fig7/results.txt
python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior train --save-recommendations results/fig7/chess_rose.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/chess_rose.pkl >> results/fig7/results.txt
python src/baselines.py --method sasrec --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --save-recommendations results/fig7/crypto_sasrec.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/crypto_sasrec.pkl >> results/fig7/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior train --save-recommendations results/fig7/crypto_rose.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/crypto_rose.pkl >> results/fig7/results.txt
python src/baselines.py --method sasrec --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --save-recommendations results/fig7/physics_sasrec.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/physics_sasrec.pkl >> results/fig7/results.txt
python src/main.py --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --prior train --save-recommendations results/fig7/physics_rose.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/physics_rose.pkl >> results/fig7/results.txt
python src/baselines.py --method sasrec --dataname math --path data/math/pretrained_seed42.pt --k 10 --save-recommendations results/fig7/math_sasrec.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/math_sasrec.pkl >> results/fig7/results.txt
python src/main.py --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --prior train --save-recommendations results/fig7/math_rose.pkl >> results/fig7/results.txt
python src/analysis.py --analysis coverage --recommendations results/fig7/math_rose.pkl >> results/fig7/results.txt
