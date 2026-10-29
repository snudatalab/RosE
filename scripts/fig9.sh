#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/fig9
: > results/fig9/results.txt

python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior train --save-recommendations results/fig9/chess.pkl >> results/fig9/results.txt
python src/analysis.py --analysis cumulative-entropy --recommendations results/fig9/chess.pkl >> results/fig9/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior train --save-recommendations results/fig9/crypto.pkl >> results/fig9/results.txt
python src/analysis.py --analysis cumulative-entropy --recommendations results/fig9/crypto.pkl >> results/fig9/results.txt
python src/main.py --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --prior train --save-recommendations results/fig9/physics.pkl >> results/fig9/results.txt
python src/analysis.py --analysis cumulative-entropy --recommendations results/fig9/physics.pkl >> results/fig9/results.txt
python src/main.py --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --prior train --save-recommendations results/fig9/math.pkl >> results/fig9/results.txt
python src/analysis.py --analysis cumulative-entropy --recommendations results/fig9/math.pkl >> results/fig9/results.txt
