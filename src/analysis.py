"""Analyze saved recommendation lists without rerunning a recommender."""

import click
import numpy as np
from scipy.sparse import csr_matrix

from experiment_util import (
    ROOT,
    bundle_diversity_gains,
    emit,
    entropy,
    forecast_remaining,
    load_recommendations,
    popularity_shift,
    spearman,
    train_test_item_popularity,
)


ANALYSES = ["approximation", "tail-frequency", "coverage",
            "popularity-shift", "segment-ndcg", "cumulative-entropy"]


def load_bundle_matrix(dataname):
    pairs = []
    itemnum = bundlenum = 0
    with (ROOT / "data" / dataname / "bundle-item.txt").open() as handle:
        for line in handle:
            bundle, item = map(int, line.rstrip().split("\t"))
            pairs.append((bundle, item))
            bundlenum, itemnum = max(bundlenum, bundle), max(itemnum, item)
    rows, cols = zip(*pairs)
    return csr_matrix((np.ones(len(pairs)), (rows, cols)),
                      shape=(bundlenum + 1, itemnum + 1))


def validate_payload(payload, bi):
    required = {"schema_version", "method", "dataname", "k", "case_count",
                "recommendations", "answers", "user_order"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"recommendation pickle is missing: {sorted(missing)}")
    recs = np.asarray(payload["recommendations"])
    answers = np.asarray(payload["answers"])
    order = np.asarray(payload["user_order"])
    count, k = int(payload["case_count"]), int(payload["k"])
    if recs.ndim != 2 or recs.shape != (count, k):
        raise ValueError("recommendation shape does not match case_count and k")
    if answers.shape != (count,) or order.shape != (count,):
        raise ValueError("answers and user_order must match case_count")
    if len(np.unique(order)) != count or np.any(order < 0):
        raise ValueError("user_order must contain unique nonnegative indices")
    if np.any(recs < 1) or np.any(recs >= bi.shape[0]):
        raise ValueError("recommendation bundle IDs are outside the catalog")
    return recs


class ApproximationAccumulator:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.absolute_error = self.relative_error = 0.0
        self.values = self.users = 0
        self.correlations = self.overlaps = 0.0

    def add(self, approximate, exact):
        error = np.abs(approximate - exact)
        self.absolute_error += error.sum()
        self.relative_error += np.sum(error / np.maximum(np.abs(exact), 1e-12))
        self.values += len(exact)
        correlation = spearman(approximate, exact)
        self.correlations += 0.0 if not np.isfinite(correlation) else correlation
        n = min(self.top_k, len(exact))
        self.overlaps += (len(set(np.argsort(-approximate)[:n]) &
                              set(np.argsort(-exact)[:n])) / n if n else 0.0)
        self.users += 1

    def result(self):
        return {
            "mae": self.absolute_error / self.values if self.values else 0.0,
            "relative_error": self.relative_error / self.values if self.values else 0.0,
            "rank_correlation": self.correlations / self.users if self.users else 0.0,
            "top_10_agreement": self.overlaps / self.users if self.users else 0.0,
        }


def approximation(recommendations, bi, payload):
    ifreq = np.zeros(bi.shape[1])
    item_prior = np.asarray(payload["item_prior"])
    k = int(payload["k"])
    horizon = payload.get("horizon")
    policy = payload.get("policy", "exact")
    accumulator = ApproximationAccumulator(top_k=10)
    for user, slate in enumerate(recommendations):
        if horizon and user and user % horizon == 0 and policy == "windowed":
            ifreq = np.zeros(bi.shape[1])
        remaining = forecast_remaining(user, len(recommendations), horizon, policy)
        expected = remaining * k * item_prior
        accumulator.add(bundle_diversity_gains(bi, ifreq, expected),
                        bundle_diversity_gains(bi, ifreq, expected, exact=True))
        for bundle in slate:
            ifreq += bi[bundle].toarray().ravel()
    return accumulator.result()


def cumulative_item_group_frequency(recommendations, bi, popularity, fraction=0.1):
    size = max(1, int(len(popularity) * fraction))
    order = np.argsort(popularity, kind="stable")
    unpopular, popular = set(order[:size] + 1), set(order[-size:] + 1)
    popular_count = unpopular_count = 0
    popular_frequency, unpopular_frequency = [], []
    for row in recommendations:
        items = np.asarray(bi[row].sum(axis=0)).ravel()
        popular_count += int(items[list(popular)].sum())
        unpopular_count += int(items[list(unpopular)].sum())
        popular_frequency.append(popular_count)
        unpopular_frequency.append(unpopular_count)
    return {"popular_frequency": popular_frequency,
            "unpopular_frequency": unpopular_frequency}


def exposure_metrics(recommendations, bi, popularity):
    tail_items = set(np.argsort(popularity)[:max(1, len(popularity) // 5)] + 1)
    coverage, tail_coverage = [], []
    for row in recommendations:
        items = set(np.unique(bi[row].indices).tolist())
        items.discard(0)
        coverage.append(len(items) / len(popularity))
        tail_coverage.append(len(items & tail_items) / len(tail_items))
    return {"item_coverage": float(np.mean(coverage)),
            "tail_coverage": float(np.mean(tail_coverage))}


def segment_ndcg(recommendations, answers, segments=10):
    values = []
    for ids in np.array_split(np.arange(len(answers)), segments):
        score = 0.0
        for index in ids:
            hits = np.flatnonzero(recommendations[index] == answers[index])
            score += 0 if not len(hits) else 1 / np.log2(hits[0] + 2)
        values.append(score / len(ids) if len(ids) else 0.0)
    return values


def cumulative_entropy(recommendations, bi, segments=10):
    points = np.array_split(np.arange(len(recommendations)), segments)
    counts, values = np.zeros(bi.shape[1]), []
    for ids in points:
        for index in ids:
            counts += np.asarray(bi[recommendations[index]].sum(axis=0)).ravel()
        values.append(entropy(counts))
    return values


@click.command()
@click.option("--analysis", type=click.Choice(ANALYSES), required=True)
@click.option("--recommendations", type=click.Path(exists=True), default=None)
@click.option("--dataname", type=click.Choice(["chess", "crypto", "physics", "math"]), default=None)
def main(analysis, recommendations, dataname):
    """Print one analysis result from one saved recommendation pickle."""
    if analysis == "popularity-shift":
        if dataname is None:
            raise click.UsageError("--dataname is required for popularity-shift")
        bi = load_bundle_matrix(dataname)
        train, test = train_test_item_popularity(dataname, bi)
        result = popularity_shift(train, test)
        result.update(analysis=analysis, dataset=dataname)
        emit(result)
        return

    if recommendations is None:
        raise click.UsageError("--recommendations is required for this analysis")
    payload = load_recommendations(recommendations)
    dataset = payload["dataname"]
    if dataname is not None and dataname != dataset:
        raise click.UsageError("--dataname does not match the recommendation pickle")
    bi = load_bundle_matrix(dataset)
    try:
        recs = validate_payload(payload, bi)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    if analysis == "approximation":
        result = approximation(recs, bi, payload)
    elif analysis == "tail-frequency":
        train, _ = train_test_item_popularity(dataset, bi)
        result = cumulative_item_group_frequency(recs, bi, train)
    elif analysis == "coverage":
        train, _ = train_test_item_popularity(dataset, bi)
        result = exposure_metrics(recs, bi, train)
    elif analysis == "segment-ndcg":
        result = {"segment_ndcg": segment_ndcg(recs, payload["answers"])}
    else:
        result = {"cumulative_entropy": cumulative_entropy(recs, bi)}
    result = {**result, "analysis": analysis, "dataset": dataset,
              "method": payload["method"],
              "source_checkpoint": payload.get("checkpoint"),
              "case_count": payload["case_count"]}
    emit(result)


if __name__ == "__main__":
    main()
