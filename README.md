# ****__RosE__****

This project is a PyTorch implementation of **__RosE__**: "Sequentially Diversified Bundle Recommendation via Rank-oriented Scoring and Entropy Estimation".

## Prerequisites
- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Scipy](https://scipy.org)
- [Click](https://click.palletsprojects.com/en/7.x/)
- [tqdm](https://tqdm.github.io/)

## Datasets
We provide four datasets in this project: Chess, Crypto, Physics, and Math.
All datasets are obtained from [StackExchange dump data](https://archive.org/details/stackexchange).
We include the preprocessed datasets in the repository: `data/{dataname}`

Math dataset is uploaded with git lfs because of its large size.
Use git lfs with the following scripts after cloning this repository to download large datasets.
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
git lfs pull
```

We also provide a [direct link](https://drive.google.com/drive/folders/1wrNShRzpG53Kn2YgnRO5jjsvFpgMYRbW?usp=sharing) to download Math dataset.
The preprocessing notebook `data/Preprocessing.ipynb` converts the original StackExchange dump data into the dataset files used by this project.

## Base Model
We provide the adapted SASRec to process sequential bundle recommendation.
The model and training code are implemented based on the code of [pmixer](https://github.com/pmixer/SASRec.pytorch).
We adopt the implementation of [rodosingh](https://github.com/pytorch/pytorch/issues/41508#issuecomment-1723119580) for a trainable Transformer layer.
The implemented models are given in `src/model.py` and pretrained parameters for each model are given in `data/{dataname}/pretrained_seed{seed}.pt`.
You may also train your own model with `src/train.py`.


## Usage

Run all commands from the repository root. RosE is evaluated with `src/main.py`:

```bash
python src/main.py \
  --dataname chess \
  --path data/chess/pretrained_seed42.pt \
  --k 10 \
  --tau 2.1
```

The command prints one operating point as a single line:

```text
method=rose dataset=chess nDCG@10=0.358 Entropy@10=5.632 Gini@10=0.387
```

Common arguments are:

- `--dataname`: dataset name (`chess`, `crypto`, `physics`, or `math`)
- `--path`: pretrained SASRec checkpoint; the default is `data/{dataname}/pretrained_seed42.pt`
- `--k`: recommendation-list length
- `--tau`: RosE accuracy-diversity trade-off parameter
- `--save-recommendations`: optional path for saving the recommendation list and its evaluation metadata as a pickle file

For example, the following command saves the recommendations to `results/chess_rose.pkl` while printing the same metric line to the terminal:

```bash
python src/main.py --dataname chess --path data/chess/pretrained_seed42.pt --k 10 --tau 2.1 --save-recommendations results/chess_rose.pkl
```

When using a custom checkpoint, the base-model arguments (`--maxlen`, `--embedding-dim`, `--num-heads`, `--dropout-rate`, and `--num-blocks`) must match the settings used to train that checkpoint.

Use the following entry points for the other methods and analyses:

- `src/main.py`: RosE
- `src/baselines.py`: SASRec and non-RosE baselines
- `src/ablations.py`: RosE ablations
- `src/analysis.py`: analyses computed from saved recommendation lists

## Reproducing the paper experiments

The `scripts/` directory contains one shell script for each table or figure. Run the corresponding script from the repository root. For example:

```bash
bash scripts/table2.sh
bash scripts/fig4.sh
```

Each script writes its terminal output to one file:

```text
results/<script-name>/results.txt
```

All experimental hyperparameters are written directly in the shell scripts. 


| Paper result | Script |
| --- | --- |
| Table 2 | `scripts/table2.sh` |
| Table 3 | `scripts/table3.sh` |
| Table 4 | `scripts/table4.sh` |
| Table 5 | `scripts/table5.sh` |
| Figure 4 | `scripts/fig4.sh` |
| Figure 5 | `scripts/fig5.sh` |
| Table 6 | `scripts/table6.sh` |
| Figure 6 | `scripts/fig6.sh` |
| Figure 7 | `scripts/fig7.sh` |
| Table 7 | `scripts/table7.sh` |
| Table 8 | `scripts/table8.sh` |
| Figure 8 | `scripts/fig8.sh` |
| Figure 9 | `scripts/fig9.sh` |
| Table 9 | `scripts/table9.sh` |
