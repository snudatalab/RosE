#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table5
: > results/table5/results.txt

python src/ablations.py --prior train --variant rose_e --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 >> results/table5/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 >> results/table5/results.txt
python src/ablations.py --prior train --variant rose_e --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 >> results/table5/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 >> results/table5/results.txt
python src/ablations.py --prior train --variant rose_e --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 >> results/table5/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 >> results/table5/results.txt
python src/ablations.py --prior train --variant rose_e --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 >> results/table5/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 >> results/table5/results.txt
