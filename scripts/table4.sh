#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table4
: > results/table4/results.txt

python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior train --save-recommendations results/table4/chess.pkl >> results/table4/results.txt
python src/analysis.py --analysis approximation --recommendations results/table4/chess.pkl >> results/table4/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior train --save-recommendations results/table4/crypto.pkl >> results/table4/results.txt
python src/analysis.py --analysis approximation --recommendations results/table4/crypto.pkl >> results/table4/results.txt
python src/main.py --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --prior train --save-recommendations results/table4/physics.pkl >> results/table4/results.txt
python src/analysis.py --analysis approximation --recommendations results/table4/physics.pkl >> results/table4/results.txt
python src/main.py --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --prior train --save-recommendations results/table4/math.pkl >> results/table4/results.txt
python src/analysis.py --analysis approximation --recommendations results/table4/math.pkl >> results/table4/results.txt
