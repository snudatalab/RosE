#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table6
: > results/table6/results.txt

python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 42 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 42 --beta 4600 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 42 --candidates 21 >> results/table6/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 42 --tau 2.1 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 43 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 43 --beta 5200 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 43 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 43 --tau 2.2 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 44 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 44 --beta 5300 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 44 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 44 --tau 2.3 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 45 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 45 --beta 5400 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 45 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 45 --tau 2.1 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 46 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 46 --beta 5000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 46 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --order-seed 46 --tau 2.2 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 42 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 42 --beta 120000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 42 --candidates 19 >> results/table6/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 42 --tau 3.1 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 43 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 43 --beta 130000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 43 --candidates 19 >> results/table6/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 43 --tau 3 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 44 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 44 --beta 130000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 44 --candidates 19 >> results/table6/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 44 --tau 2.9 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 45 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 45 --beta 130000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 45 --candidates 19 >> results/table6/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 45 --tau 3.1 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 46 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 46 --beta 130000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 46 --candidates 19 >> results/table6/results.txt
python src/main.py --prior train --dataname crypto --path data/crypto/pretrained_seed42.pt --k 10 --order-seed 46 --tau 2.9 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 42 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 42 --beta 12000000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 42 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 42 --tau 4.9 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 43 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 43 --beta 12000000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 43 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 43 --tau 5 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 44 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 44 --beta 12000000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 44 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 44 --tau 5.1 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 45 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 45 --beta 12000000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 45 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 45 --tau 5 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 46 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 46 --beta 12000000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 46 --candidates 22 >> results/table6/results.txt
python src/main.py --prior train --dataname physics --path data/physics/pretrained_seed42.pt --k 10 --order-seed 46 --tau 5 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 42 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 42 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 42 --beta 440000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 42 --candidates 18 >> results/table6/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 42 --tau 3.8 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 43 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 43 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 43 --beta 440000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 43 --candidates 18 >> results/table6/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 43 --tau 3.9 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 44 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 44 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 44 --beta 440000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 44 --candidates 18 >> results/table6/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 44 --tau 3.9 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 45 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 45 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 45 --beta 440000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 45 --candidates 18 >> results/table6/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 45 --tau 3.9 >> results/table6/results.txt
python src/baselines.py --method kwon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 46 --threshold 0.001 >> results/table6/results.txt
python src/baselines.py --method dpp --epsilon 1e-10 --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method uimatch --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 46 >> results/table6/results.txt
python src/baselines.py --method popcon --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 46 --beta 440000 --candidates 100 >> results/table6/results.txt
python src/baselines.py --method sapid --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 46 --candidates 18 >> results/table6/results.txt
python src/main.py --prior train --dataname math --path data/math/pretrained_seed42.pt --k 10 --order-seed 46 --tau 3.9 >> results/table6/results.txt
