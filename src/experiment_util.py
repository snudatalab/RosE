"""Shared data, metric, serialization, and analysis utilities."""

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import entr
from scipy.spatial.distance import jensenshannon

from main_util import eval as evaluate
from main_util import test_data
from model import SBR


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_identity(dataname, path=None):
    return str(Path(path) if path else Path("data") / dataname / "pretrained_seed42.pt")


def load_experiment(dataname, path=None, maxlen=200, embedding_dim=128,
                    num_heads=1, dropout_rate=0.5, num_blocks=2):
    data_dir = ROOT / "data" / dataname
    bundles = defaultdict(list)
    item_num = 0
    with (data_dir / "bundle-item.txt").open() as handle:
        for line in handle:
            bundle, item = map(int, line.rstrip().split("\t"))
            bundles[bundle].append(item)
            item_num = max(item_num, item)
    pairs = [(bundle, item) for bundle, items in bundles.items() for item in items]
    rows, cols = zip(*pairs)
    bi = csr_matrix((np.ones(len(pairs)), (rows, cols)),
                    shape=(max(bundles) + 1, item_num + 1))

    old_cwd = Path.cwd()
    try:
        os.chdir(Path(__file__).parent)
        _, bundlenum, inputs, answers, bprob = test_data(dataname, maxlen)
    finally:
        os.chdir(old_cwd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SBR(bundlenum, item_num, bundles, max(map(len, bundles.values())), device,
                maxlen=maxlen, embedding_dim=embedding_dim, num_heads=num_heads,
                dropout_rate=dropout_rate, num_blocks=num_blocks).to(device)
    checkpoint = Path(path) if path else data_dir / "pretrained_seed42.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, inputs, np.asarray(answers), bprob, bi, bundlenum, item_num


def reorder_users(inputs, answers, order_seed=None, max_users=None,
                  source_order=None):
    order = np.arange(len(inputs))
    if order_seed is not None:
        base = order if source_order is None else np.asarray(source_order)
        order = base[np.random.default_rng(order_seed).permutation(len(base))]
    if max_users is not None:
        order = order[:max_users]
    return [inputs[i] for i in order], np.asarray(answers)[order], order


def stable_preference(scores, tau=1.0):
    scores = np.asarray(scores, dtype=float)
    values = np.exp((scores - scores.max()) / tau)
    span = values.max() - values.min()
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def entropy(counts):
    counts = np.asarray(counts, dtype=float)[1:]
    total = counts.sum()
    return 0.0 if total == 0 else float(entr(counts / total).sum())


def approximate_item_gain(expected_counts):
    """Independent one-item Shannon entropy gains; index zero is padding."""
    counts = np.asarray(expected_counts, dtype=float)
    gain = np.zeros_like(counts)
    total = counts[1:].sum()
    if total <= 0:
        return gain
    base = entropy(counts)
    xlog = np.zeros_like(counts)
    positive = counts > 0
    xlog[positive] = counts[positive] * np.log(counts[positive])
    plus = counts + 1
    delta = plus * np.log(plus) - xlog
    gain[1:] = (np.log(total + 1) -
                (xlog[1:].sum() + delta[1:]) / (total + 1) - base)
    return gain


def bundle_diversity_gains(bi, ifreq, expected, exact=False):
    future = np.asarray(ifreq, dtype=float) + np.asarray(expected, dtype=float)
    item_gain = approximate_item_gain(future)
    max_size = max(1.0, float(np.asarray(bi.sum(axis=1)).max()))
    scale = max_size * float(np.abs(item_gain[1:]).max(initial=0))
    if exact:
        counts = future[1:]
        total = counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            xlogx = np.where(counts > 0, counts * np.log(counts), 0.0)
            increment = (counts + 1) * np.log(counts + 1) - xlogx
        sizes = np.asarray(bi[1:, 1:].sum(axis=1)).ravel()
        totals = total + sizes
        new_xlogx = xlogx.sum() + np.asarray(bi[1:, 1:].dot(increment)).ravel()
        base = np.log(total) - xlogx.sum() / total if total > 0 else 0.0
        gains = np.zeros_like(totals, dtype=float)
        valid = totals > 0
        gains[valid] = np.log(totals[valid]) - new_xlogx[valid] / totals[valid] - base
    else:
        gains = np.asarray(bi[1:].dot(item_gain)).ravel()
    return gains / scale if scale else np.zeros_like(gains)


def histories(dataname):
    result = defaultdict(list)
    with (ROOT / "data" / dataname / "user-bundle.txt").open() as handle:
        for line in handle:
            user, bundle, timestamp = map(int, line.rstrip().split("\t"))
            result[user].append((timestamp, bundle))
    return result


def source_order_indices(dataname):
    """Map source-file user order to the chronological public-runner rows."""
    user_histories = histories(dataname)
    source_users = list(user_histories)
    chronological_users = sorted(
        source_users,
        key=lambda user: (max(timestamp for timestamp, _ in user_histories[user]),
                          user),
    )
    chronological_index = {user: index for index, user in enumerate(chronological_users)}
    return np.asarray([chronological_index[user] for user in source_users])


def training_item_prior(dataname, bi):
    counts = np.zeros(bi.shape[1])
    observations = 0
    for history in histories(dataname).values():
        bundles = [bundle for _, bundle in sorted(history)]
        train_bundles = bundles[:-2] if len(bundles) >= 3 else bundles
        if train_bundles:
            counts += np.asarray(bi[train_bundles].sum(axis=0)).ravel()
            observations += len(train_bundles)
    return counts / observations if observations else counts


def uniform_item_prior(bi):
    bundle_count = bi.shape[0] - 1
    counts = np.asarray(bi[1:].sum(axis=0)).ravel()
    return counts / bundle_count if bundle_count else counts


def validation_sequences(dataname, maxlen=200):
    sequences = []
    for history in histories(dataname).values():
        bundles = [bundle for _, bundle in sorted(history)]
        if len(bundles) < 2:
            continue
        sequence = np.zeros(maxlen, dtype=int)
        prefix = bundles[:-2][-maxlen:]
        if prefix:
            sequence[-len(prefix):] = prefix
        sequences.append(sequence)
    return sequences


def recommendation_item_prior(recommendations, bi):
    counts = np.zeros(bi.shape[1])
    observations = 0
    for slate in recommendations:
        counts += np.asarray(bi[np.asarray(slate, dtype=int)].sum(axis=0)).ravel()
        observations += len(slate)
    return counts / observations if observations else counts


def make_item_prior(name, dataname, bi, bprob, model=None, k=10, maxlen=200):
    if name == "legacy":
        return np.asarray(bi.T.dot(bprob)).ravel()
    if name == "train":
        return training_item_prior(dataname, bi)
    if name == "uniform":
        return uniform_item_prior(bi)
    if name == "sasrec":
        recommendations = []
        with torch.no_grad():
            for sequence in validation_sequences(dataname, maxlen):
                score = model.predict(torch.as_tensor(sequence, dtype=torch.long,
                                                      device=model.device))
                recommendations.append(np.argsort(-score.squeeze().cpu().numpy())[:k] + 1)
        return recommendation_item_prior(recommendations, bi)
    raise ValueError(f"unknown prior: {name}")


def forecast_remaining(user, total_users, horizon, policy):
    if policy == "exact":
        return total_users - user - 1
    if horizon is None:
        raise ValueError("fixed-horizon policies require --horizon")
    return horizon - (user % horizon)


def metrics(method, dataname, k, recs, answers, bi, bundlenum, itemnum):
    ndcg, diversity, gini = evaluate(torch.as_tensor(recs), answers, bi,
                                     bundlenum, itemnum, k)
    return {"method": method, "dataset": dataname, "k": k,
            "ndcg": float(ndcg), "entropy": float(diversity), "gini": float(gini)}


def emit(result):
    performance_keys = {"method", "dataset", "k", "ndcg", "entropy", "gini"}
    if performance_keys.issubset(result):
        k = result["k"]
        print(
            f"method={result['method']} dataset={result['dataset']} "
            f"nDCG@{k}={result['ndcg']:.3f} "
            f"Entropy@{k}={result['entropy']:.3f} "
            f"Gini@{k}={result['gini']:.3f}"
        )
        return
    hidden = {"source_checkpoint", "case_count"}
    print(" ".join(f"{key}={value}" for key, value in result.items()
                   if key not in hidden))


def save_recommendations(path, recommendations, answers, order, **metadata):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    recommendations = np.asarray(recommendations)
    payload = dict(metadata)
    payload.update(schema_version=1, case_count=len(recommendations),
                   recommendations=recommendations,
                   answers=np.asarray(answers), user_order=np.asarray(order))
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


class _RecommendationUnpickler(pickle.Unpickler):
    """Load only the NumPy types written by save_recommendations()."""

    _NUMPY_TYPES = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module, name):
        if (module, name) in self._NUMPY_TYPES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"unsupported pickle type: {module}.{name}")


def load_recommendations(path):
    with Path(path).open("rb") as handle:
        payload = _RecommendationUnpickler(handle).load()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported recommendation pickle schema")
    return payload


def spearman(left, right):
    def ranks(values):
        values = np.asarray(values)
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result
    a, b = ranks(left), ranks(right)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def train_test_item_popularity(dataname, bi):
    train = np.zeros(bi.shape[1])
    test = np.zeros(bi.shape[1])
    for history in histories(dataname).values():
        bundles = [bundle for _, bundle in sorted(history)]
        if len(bundles) >= 3:
            train_bundles, test_bundle = bundles[:-2], bundles[-1]
        else:
            train_bundles, test_bundle = bundles, None
        if train_bundles:
            train += np.asarray(bi[train_bundles].sum(axis=0)).ravel()
        if test_bundle is not None:
            test += bi[test_bundle].toarray().ravel()
    return train[1:], test[1:]


def popularity_shift(train, test):
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    p, q = train / train.sum(), test / test.sum()
    rank_train_values = np.r_[0.0, train]
    rank_test_values = np.r_[0.0, test]
    train_rank = np.empty(len(rank_train_values), dtype=float)
    test_rank = np.empty(len(rank_test_values), dtype=float)
    train_rank[np.argsort(rank_train_values, kind="stable")] = np.arange(len(rank_train_values))
    test_rank[np.argsort(rank_test_values, kind="stable")] = np.arange(len(rank_test_values))
    train_rank -= train_rank.mean()
    test_rank -= test_rank.mean()
    correlation = np.dot(train_rank, test_rank) / np.sqrt(
        np.dot(train_rank, train_rank) * np.dot(test_rank, test_rank))
    return {"js_divergence": float(jensenshannon(p, q, base=2) ** 2),
            "rank_correlation": float(correlation)}
