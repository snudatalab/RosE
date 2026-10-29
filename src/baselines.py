"""Run one non-RosE baseline operating point."""

import click
import numpy as np
import torch

from experiment_util import (
    checkpoint_identity,
    emit,
    entropy,
    load_experiment,
    metrics,
    reorder_users,
    save_recommendations as save_pickle,
    source_order_indices,
)


def _entropy_after_unit_increments(counts, indices):
    """Entropy after adding one count to each listed item."""
    counts = np.asarray(counts, dtype=float)
    indices = np.unique(np.asarray(indices, dtype=int))
    total = float(counts.sum())
    positive = counts[counts > 0]
    xlogx = float((positive * np.log(positive)).sum())
    old = counts[indices]
    new = old + 1
    old_terms = np.zeros_like(old)
    mask = old > 0
    old_terms[mask] = old[mask] * np.log(old[mask])
    xlogx += float((new * np.log(new) - old_terms).sum())
    new_total = total + len(indices)
    return float(np.log(new_total) - xlogx / new_total)


def _sapid_select(pool, expected, k):
    """Preserve the published NumPy default tie ordering."""
    picked = np.argsort(expected[pool])[:k]
    return pool[np.sort(picked)]


def dpp(scores, features, k, epsilon=1e-10):
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms == 0, 1, norms)
    cis = np.zeros((k, len(scores)))
    residual = scores * scores
    selected = int(np.argmax(residual))
    chosen = [selected]
    while len(chosen) < k:
        row = len(chosen) - 1
        denominator = np.sqrt(max(residual[selected], np.finfo(float).eps))
        kernel_row = scores[selected] * scores * np.sum(
            features * features[selected], axis=1)
        projection = (np.sum(cis[:row] * cis[:row, selected, None], axis=0)
                      if row else 0.0)
        eis = (kernel_row - projection) / denominator
        cis[row] = eis
        residual -= eis * eis
        residual[chosen] = -np.inf
        selected = int(np.argmax(residual))
        if not np.isfinite(residual[selected]) or residual[selected] < epsilon:
            break
        chosen.append(selected)
    if len(chosen) < k:
        chosen.extend(i for i in np.argsort(-scores) if i not in chosen)
    return np.asarray(chosen[:k]) + 1


def recommend(method, scores, bi, state, k, remaining, bprob,
              beta=10000.0, candidates=20, features=None,
              threshold=0.001, epsilon=1e-10):
    """Select one slate using only the requested baseline."""
    scores = np.asarray(scores, dtype=float)
    candidates = max(k, min(candidates, len(scores)))
    pref_order = np.argsort(-scores)

    if method == "sasrec":
        selected = pref_order[:k] + 1
    elif method == "kwon":
        span = scores.max() - scores.min()
        norm = np.zeros_like(scores) if span == 0 else (scores - scores.min()) / span
        cutoff, relaxed = 1 - threshold, 1 - 0.5 * threshold
        # Preserve the published notebook's range(item_num) loop bound even
        # though these are bundle scores. This legacy quirk is part of the
        # reported Kwon baseline semantics.
        legacy_limit = min(bi.shape[1] - 1, len(scores))
        eligible = np.flatnonzero(norm[:legacy_limit] > cutoff)
        near = norm[:legacy_limit][norm[:legacy_limit] > relaxed]
        alpha = near.max(initial=0)
        ranks = np.full(len(scores), 1000.0)
        for index in eligible:
            ranks[index] = norm[index] if norm[index] >= relaxed else alpha + 1 / norm[index]
        selected = np.argsort(ranks, kind="stable")[:k] + 1
    elif method == "dpp":
        selected = dpp(scores, features, k, epsilon)
    elif method == "uimatch":
        left_slot = state["left_slot"]
        selected = []
        for index in pref_order + 1:
            if left_slot[index] > 0:
                selected.append(index)
                left_slot[index] -= 1
            if len(selected) == k:
                break
        if len(selected) < k:
            left_slot += 1
            for index in pref_order + 1:
                if left_slot[index] > 0 and index not in selected:
                    selected.append(index)
                    left_slot[index] -= 1
                if len(selected) == k:
                    break
        selected = np.asarray(selected)
    elif method == "sapid":
        ranked = pref_order[:candidates] + 1
        ranked_scores = scores[pref_order[:candidates]]
        gaps = ranked_scores[:-1] - ranked_scores[1:]
        if len(gaps):
            large = np.flatnonzero(gaps > np.percentile(gaps, 75))
            cut = large[-1] if len(large) else k
        else:
            cut = k
        pool = ranked[:cut] if cut > k else ranked[:k]
        expected = state["bfreq"] + remaining * k * bprob
        selected = _sapid_select(pool, expected, k)
    else:  # PopCon
        if state["ifreq"].sum() == 0:
            selected = pref_order[:k] + 1
        else:
            pool0 = pref_order[:candidates]
            pool = pool0 + 1
            base = entropy(state["ifreq"])
            coverage = np.array([(state["ifreq"][bi[b].indices] == 0).sum()
                                 for b in pool]) / max(1, bi.shape[1] - 1)
            gains = np.array([
                _entropy_after_unit_increments(state["ifreq"], bi[b].indices) - base
                for b in pool
            ])
            diversity = 0.5 * coverage + 0.5 * gains / max(1.0, np.log2(bi.shape[1] - 1))
            accuracy = 1 / (1 + np.exp(-scores[pool0]))
            combined = accuracy ** beta + (1 - accuracy ** beta) * diversity
            selected = pool[np.argsort(-combined)[:k]]

    for bundle in selected:
        state["bfreq"][bundle] += 1
        state["ifreq"] += bi[bundle].toarray().ravel()
    return np.asarray(selected, dtype=int)


@click.command()
@click.option("--method", type=click.Choice(["sasrec", "kwon", "dpp", "uimatch", "popcon", "sapid"]), required=True)
@click.option("--dataname", type=click.Choice(["chess", "crypto", "physics", "math"]), default="chess")
@click.option("--path", type=click.Path(), default=None)
@click.option("--k", default=10, type=int)
@click.option("--beta", default=10000.0, type=float)
@click.option("--candidates", default=20, type=int)
@click.option("--threshold", default=0.001, type=float)
@click.option("--epsilon", default=1e-10, type=float)
@click.option("--order-seed", default=None, type=int)
@click.option("--max-users", default=None, type=click.IntRange(min=1))
@click.option("--save-recommendations", type=click.Path(), default=None)
def main(method, dataname, path, k, beta, candidates, threshold, epsilon,
         order_seed, max_users, save_recommendations):
    """Run and print one baseline operating point as one terminal line."""
    model, inputs, answers, bprob, bi, bundlenum, itemnum = load_experiment(dataname, path)
    inputs, answers, order = reorder_users(
        inputs, answers, order_seed, max_users, source_order_indices(dataname))
    state = {"ifreq": np.zeros(itemnum + 1), "bfreq": np.zeros(bundlenum + 1)}
    if method == "uimatch":
        state["left_slot"] = np.ceil(bprob * k * len(inputs))
    features = None
    if method == "dpp":
        with torch.no_grad():
            ids = torch.arange(1, bundlenum + 1, device=model.device)
            features = model.sess2embs(ids).cpu().numpy()

    rows = []
    with torch.no_grad():
        for user, sequence in enumerate(inputs):
            score = model.predict(torch.as_tensor(sequence, dtype=torch.long,
                                                  device=model.device))
            rows.append(recommend(method, score.squeeze().cpu().numpy(), bi, state,
                                  k, len(inputs) - user - 1, bprob, beta,
                                  candidates, features, threshold, epsilon))
    recommendations = np.asarray(rows)
    result = metrics(method, dataname, k, recommendations, answers, bi,
                     bundlenum, itemnum)
    result.update(order_seed=order_seed, beta=beta, candidates=candidates,
                  threshold=threshold, epsilon=epsilon,
                  checkpoint=checkpoint_identity(dataname, path),
                  case_count=len(recommendations), max_users=max_users)
    save_pickle(save_recommendations, recommendations, answers, order,
                method=method, dataname=dataname, k=k, order_seed=order_seed,
                beta=beta, candidates=candidates, threshold=threshold,
                epsilon=epsilon, checkpoint=checkpoint_identity(dataname, path),
                max_users=max_users)
    emit(result)


if __name__ == "__main__":
    main()
