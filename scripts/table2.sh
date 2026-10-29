#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table2
: > results/table2/results.txt

# Chess - legacy pretrained_seed42.pt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 5 --beta 8800 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 5 --candidates 12 >> results/table2/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 5 --tau 1.6 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --beta 4700 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --candidates 21 >> results/table2/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 >> results/table2/results.txt

# Chess - finite-attention-v2 seed 43
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed43.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed43.pt --k 5 --beta 2200 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed43.pt --k 5 --candidates 15 >> results/table2/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed43.pt --k 5 --tau 0.66 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --beta 1700 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --candidates 24 >> results/table2/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed43.pt --k 10 --tau 0.83 >> results/table2/results.txt

# Chess - finite-attention-v2 seed 44
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed44.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed44.pt --k 5 --beta 1900 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed44.pt --k 5 --candidates 18 >> results/table2/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed44.pt --k 5 --tau 0.55 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --beta 1300 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --candidates 27 >> results/table2/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed44.pt --k 10 --tau 0.85 >> results/table2/results.txt

# Crypto - legacy pretrained_seed42.pt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 5 --beta 310000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 5 --candidates 12 >> results/table2/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 5 --tau 2.5 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --beta 120000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --candidates 19 >> results/table2/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --tau 3 >> results/table2/results.txt

# Crypto - finite-attention-v2 seed 43
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 5 --beta 9200 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed43.pt --k 5 --candidates 15 >> results/table2/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed43.pt --k 5 --tau 0.72 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --beta 7400 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --candidates 23 >> results/table2/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed43.pt --k 10 --tau 0.89 >> results/table2/results.txt

# Crypto - finite-attention-v2 seed 44
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 5 --beta 6600 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed44.pt --k 5 --candidates 18 >> results/table2/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed44.pt --k 5 --tau 0.84 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --beta 5400 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --candidates 23 >> results/table2/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed44.pt --k 10 --tau 1.1 >> results/table2/results.txt

# Physics - legacy pretrained_seed42.pt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 5 --beta 29000000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 5 --candidates 14 >> results/table2/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 5 --tau 4.3 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --beta 13000000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --candidates 21 >> results/table2/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --tau 5 >> results/table2/results.txt

# Physics - finite-attention-v2 seed 43
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed43.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed43.pt --k 5 --beta 1500000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed43.pt --k 5 --candidates 22 >> results/table2/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed43.pt --k 5 --tau 0.62 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --beta 1300000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --candidates 30 >> results/table2/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed43.pt --k 10 --tau 0.92 >> results/table2/results.txt

# Physics - finite-attention-v2 seed 44
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed44.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed44.pt --k 5 --beta 1400000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed44.pt --k 5 --candidates 19 >> results/table2/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed44.pt --k 5 --tau 0.51 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --beta 1300000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --candidates 30 >> results/table2/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed44.pt --k 10 --tau 0.77 >> results/table2/results.txt

# Math - legacy pretrained_seed42.pt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 5 --beta 620000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 5 --candidates 11 >> results/table2/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 5 --tau 3 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --beta 420000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --candidates 18 >> results/table2/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --tau 4 >> results/table2/results.txt

# Math - finite-attention-v2 seed 43
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed43.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed43.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed43.pt --k 5 --beta 800000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed43.pt --k 5 --candidates 14 >> results/table2/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed43.pt --k 5 --tau 2 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed43.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed43.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed43.pt --k 10 --beta 600000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed43.pt --k 10 --candidates 20 >> results/table2/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed43.pt --k 10 --tau 2.2 >> results/table2/results.txt

# Math - finite-attention-v2 seed 44
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed44.pt --k 5 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed44.pt --k 5 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed44.pt --k 5 --beta 800000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed44.pt --k 5 --candidates 14 >> results/table2/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed44.pt --k 5 --tau 2 >> results/table2/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed44.pt --k 10 --threshold 0.001 >> results/table2/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed44.pt --k 10 >> results/table2/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed44.pt --k 10 --beta 600000 --candidates 100 >> results/table2/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed44.pt --k 10 --candidates 20 >> results/table2/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed44.pt --k 10 --tau 2.2 >> results/table2/results.txt
