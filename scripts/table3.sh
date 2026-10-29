#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table3
: > results/table3/results.txt

python src/ablations.py --prior train --variant rose_a --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --gamma 27 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_d --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 1.6 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_j --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 1.6 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --beta 50 --tau 2.1 >> results/table3/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_a --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --gamma 25 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_d --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 2.3 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_j --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 2.4 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --beta 42 --tau 3 >> results/table3/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_a --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --gamma 15 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_d --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 3.7 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_j --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 4 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --beta 41 --tau 5 >> results/table3/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_a --dataname math --path data/math/pretrained_seed42.pt --k 10 --gamma 12 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_d --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 2.6 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_j --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 2.9 >> results/table3/results.txt
python src/ablations.py --prior train --variant rose_popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --beta 15 --tau 4 >> results/table3/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 >> results/table3/results.txt
