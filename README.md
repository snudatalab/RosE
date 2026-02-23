# ****__RosE__****

This project is a PyTorch implementation of **__RosE__**: "Accurate and Diversified Sequential Bundle Recommendation".

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
They are uploaded with git lfs due to their large size.
Use git lfs with the following scripts after cloning this repository to download large datasets.
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
git lfs pull
```

We also provide a [direct link](https://drive.google.com/drive/folders/1gr95tP3Ae7f4c4V5mx5jzLETHDWdVis1?usp=drive_link) to download all datasets. 

## Base Model
We provide the adapted SASRec to process sequential bundle recommendation.
The model and training code are implemented based on the code of [pmixer](https://github.com/pmixer/SASRec.pytorch).
We adopt the implementation of [rodosingh](https://github.com/pytorch/pytorch/issues/41508#issuecomment-1723119580) for a trainable Transformer layer.
The implemented models are given in `model.py` and pretrained parameters for each model are given in `data/{dataname}/pretrained.pt`.
You may also train your own model with `training.py`.


## Usage

You can run the evaluation code with the command `python main.py` with the following arguments.
* Specify the dataset and the base model
  * `dataname` : choose a dataset (default: Chess)
  * `path` : choose a pretrained parameters (default: `data/{dataname}/pretrained`)
* Hyperparameters of the base model (Must be identical to the hyperparameters selected to pretrain the model)
  * `maxlen` : the maximum length of the bundle sequence
  * `embedding_dim` : the length of embeddings for bundles and items
  * `num_heads` : the number of attention heads
  * `dropout_rate` : the dropout ratio
  * `num_blocks` : the number of Transformer layers
* Hyperparameter of **__RosE__**
  * `tau` : control the trade-off between accuracy and diversity
* Specify the length of recommendation list
  * `k` : the length of recommendation list

As default, `python main.py` would reproduce the results reported in the paper for Chess dataset.