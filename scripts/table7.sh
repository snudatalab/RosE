#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table7
: > results/table7/results.txt

python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior train >> results/table7/results.txt
python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior uniform >> results/table7/results.txt
python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --prior sasrec >> results/table7/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior train >> results/table7/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior uniform >> results/table7/results.txt
python src/main.py --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --prior sasrec >> results/table7/results.txt
python src/main.py --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --prior train >> results/table7/results.txt
python src/main.py --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --prior uniform >> results/table7/results.txt
python src/main.py --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --prior sasrec >> results/table7/results.txt
python src/main.py --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --prior train >> results/table7/results.txt
python src/main.py --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --prior uniform >> results/table7/results.txt
python src/main.py --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --prior sasrec >> results/table7/results.txt
