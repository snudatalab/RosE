"""Run one RosE ablation operating point."""

import click
import numpy as np
import torch

from experiment_util import (
    bundle_diversity_gains,
    checkpoint_identity,
    emit,
    load_experiment,
    make_item_prior,
    metrics,
    reorder_users,
    source_order_indices,
    stable_preference,
)


def recommend(variant, scores, bi, ifreq, k, expected_items,
              tau=1.0, beta=50.0, gamma=1.0):
    exact = variant == "rose_e"
    diversity = bundle_diversity_gains(bi, ifreq, expected_items, exact=exact)

    if variant == "rose_d":
        future = ifreq + expected_items
        item_gain = np.median(future[1:]) - future
        item_gain[0] = 0
        scale = np.max(future[1:], initial=0)
        if scale:
            item_gain /= scale
        max_size = max(1.0, float(np.asarray(bi.sum(axis=1)).max()))
        diversity = np.asarray(bi[1:].dot(item_gain)).ravel() / max_size

    if variant == "rose_a":
        span = scores.max() - scores.min()
        accuracy = np.zeros_like(scores) if span == 0 else (scores - scores.min()) / span
    else:
        accuracy = stable_preference(scores, tau)

    if variant == "rose_j":
        combined = 2 * accuracy + diversity
    elif variant == "rose_popcon":
        scaled = 1 / (1 + np.exp(-accuracy))
        combined = scaled ** beta + (1 - scaled ** beta) * diversity
    else:
        combined = accuracy + (1 - accuracy) * diversity * gamma
    selected = np.argsort(-combined)[:k] + 1
    return selected[np.argsort(-scores[selected - 1])]


@click.command()
@click.option("--variant", type=click.Choice(["rose_a", "rose_d", "rose_j", "rose_popcon", "rose_e"]), required=True)
@click.option("--dataname", type=click.Choice(["chess", "crypto", "physics", "math"]), default="chess")
@click.option("--path", type=click.Path(), default=None)
@click.option("--k", default=10, type=int)
@click.option("--tau", default=2.1, type=float)
@click.option("--beta", default=50.0, type=float)
@click.option("--gamma", default=1.0, type=float)
@click.option("--prior", type=click.Choice(["legacy", "train", "uniform", "sasrec"]), default="train")
@click.option("--max-users", default=None, type=click.IntRange(min=1))
def main(variant, dataname, path, k, tau, beta, gamma, prior, max_users):
    """Run and print one ablation operating point as one terminal line."""
    model, inputs, answers, bprob, bi, bundlenum, itemnum = load_experiment(dataname, path)
    inputs, answers, _ = reorder_users(
        inputs, answers, max_users=max_users,
        source_order=source_order_indices(dataname))
    item_prior = make_item_prior(prior, dataname, bi, bprob, model, k)
    ifreq = np.zeros(itemnum + 1)
    rows = []
    with torch.no_grad():
        for user, sequence in enumerate(inputs):
            score = model.predict(torch.as_tensor(sequence, dtype=torch.long,
                                                  device=model.device))
            score = score.squeeze().cpu().numpy()
            expected = (len(inputs) - user - 1) * k * item_prior
            selected = recommend(variant, score, bi, ifreq, k, expected,
                                 tau=tau, beta=beta, gamma=gamma)
            for bundle in selected:
                ifreq += bi[bundle].toarray().ravel()
            rows.append(selected)
    recommendations = np.asarray(rows)
    result = metrics(variant, dataname, k, recommendations, answers, bi,
                     bundlenum, itemnum)
    result.update(tau=tau, beta=beta, gamma=gamma, prior=prior,
                  checkpoint=checkpoint_identity(dataname, path),
                  case_count=len(recommendations), max_users=max_users)
    emit(result)


if __name__ == "__main__":
    main()
