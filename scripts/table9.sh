#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table9
: > results/table9/results.txt

python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy exact >> results/table9/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy windowed --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy windowed --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy windowed --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy retained --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy retained --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --policy retained --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy exact >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy windowed --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy windowed --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy windowed --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy retained --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy retained --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 --policy retained --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy exact >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy windowed --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy windowed --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy windowed --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy retained --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy retained --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 --policy retained --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy exact >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy windowed --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy windowed --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy windowed --horizon 1000 >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy retained --horizon 100 >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy retained --horizon 500 >> results/table9/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 --policy retained --horizon 1000 >> results/table9/results.txt
