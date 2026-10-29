#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results/table8
: > results/table8/results.txt

python src/analysis.py --analysis popularity-shift --dataname chess >> results/table8/results.txt
python src/analysis.py --analysis popularity-shift --dataname crypto >> results/table8/results.txt
python src/analysis.py --analysis popularity-shift --dataname physics >> results/table8/results.txt
python src/analysis.py --analysis popularity-shift --dataname math >> results/table8/results.txt
