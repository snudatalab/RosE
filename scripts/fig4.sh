#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/fig4
: > results/fig4/results.txt

# Chess - legacy pretrained_seed42.pt
python src/baselines.py --method sasrec --dataname chess --path data/chess/pretrained_seed42.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --beta 100 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --beta 1000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Chess - finite-attention-v2 seed 43
python src/baselines.py --method sasrec --dataname chess --path data/chess/pretrained_seed43.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --beta 100 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --beta 1000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Chess - finite-attention-v2 seed 44
python src/baselines.py --method sasrec --dataname chess --path data/chess/pretrained_seed44.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --beta 100 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --beta 1000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Crypto - legacy pretrained_seed42.pt
python src/baselines.py --method sasrec --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Crypto - finite-attention-v2 seed 43
python src/baselines.py --method sasrec --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Crypto - finite-attention-v2 seed 44
python src/baselines.py --method sasrec --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Physics - legacy pretrained_seed42.pt
python src/baselines.py --method sasrec --dataname physics --path data/physics/pretrained_seed42.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --beta 100000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --beta 1000000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Physics - finite-attention-v2 seed 43
python src/baselines.py --method sasrec --dataname physics --path data/physics/pretrained_seed43.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --beta 100000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --beta 1000000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Physics - finite-attention-v2 seed 44
python src/baselines.py --method sasrec --dataname physics --path data/physics/pretrained_seed44.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --beta 100000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --beta 1000000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Math - legacy pretrained_seed42.pt
python src/baselines.py --method sasrec --dataname math --path data/math/pretrained_seed42.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Math - finite-attention-v2 seed 43
python src/baselines.py --method sasrec --dataname math --path data/math/pretrained_seed43.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed43.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed43.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed43.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed43.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed43.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed43.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed43.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed43.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed43.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed43.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed43.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed43.pt --k 10 --candidates 45 >> results/fig4/results.txt

# Math - finite-attention-v2 seed 44
python src/baselines.py --method sasrec --dataname math --path data/math/pretrained_seed44.pt --k 10 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed44.pt --k 10 --tau 1 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed44.pt --k 10 --tau 3 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed44.pt --k 10 --tau 5 >> results/fig4/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed44.pt --k 10 --tau 10 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed44.pt --k 10 --beta 10000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed44.pt --k 10 --beta 100000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed44.pt --k 10 --beta 1000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed44.pt --k 10 --beta 10000000 --candidates 100 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed44.pt --k 10 --candidates 15 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed44.pt --k 10 --candidates 25 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed44.pt --k 10 --candidates 35 >> results/fig4/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed44.pt --k 10 --candidates 45 >> results/fig4/results.txt
