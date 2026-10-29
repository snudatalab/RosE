"""Run one RosE operating point."""

import click
import numpy as np
import torch

from experiment_util import (
    bundle_diversity_gains,
    checkpoint_identity,
    emit,
    forecast_remaining,
    load_experiment,
    make_item_prior,
    metrics,
    reorder_users,
    save_recommendations as save_pickle,
    source_order_indices,
)
from main_util import min_max_normalize, softmax


def RosE(model, test_input, iprob, itemnum, max_bundle_size, BIsparse,
         tau=1, k=10, horizon=None, policy="exact"):
    """Generate RosE recommendations in chronological user order."""
    newreclist = []
    ifreq = np.zeros(itemnum + 1)

    with torch.no_grad():
        for u, sequence in enumerate(test_input):
            if horizon and u and u % horizon == 0 and policy == "windowed":
                ifreq = np.zeros(itemnum + 1)

            pred = model.predict(torch.as_tensor(sequence, dtype=torch.long,
                                                 device=model.device))
            pred = pred.squeeze().cpu().numpy()
            bpref = np.insert(pred, 0, 0)

            remaining = forecast_remaining(u, len(test_input), horizon, policy)
            expected = remaining * k * iprob
            # Item gains are normalized before their linear bundle aggregation.
            # This is equivalent to scaling afterward and keeps values small.
            divgain = bundle_diversity_gains(BIsparse, ifreq, expected)

            score = min_max_normalize(softmax(bpref[1:], temperature=tau))
            joint = score + (1 - score) * divgain
            reclist = (-joint).argsort()[:k] + 1
            reclist = sorted(reclist, reverse=True, key=lambda b: bpref[b])

            for bundle in reclist:
                ifreq += BIsparse[bundle].toarray().squeeze()
            newreclist.append(reclist)

    return torch.as_tensor(newreclist, dtype=torch.long)


@click.command()
@click.option("--maxlen", default=200, type=int)
@click.option("--embedding-dim", "embedding_dim", default=128, type=int)
@click.option("--num-heads", "num_heads", default=1, type=int)
@click.option("--dropout-rate", "dropout_rate", default=0.5, type=float)
@click.option("--num-blocks", "num_blocks", default=2, type=int)
@click.option("--dataname", type=click.Choice(["chess", "crypto", "physics", "math"]), default="chess")
@click.option("--path", type=click.Path(), default=None)
@click.option("--tau", default=2.1, type=float)
@click.option("--k", default=10, type=int)
@click.option("--prior", type=click.Choice(["legacy", "train", "uniform", "sasrec"]), default="train")
@click.option("--order-seed", default=None, type=int)
@click.option("--policy", type=click.Choice(["exact", "windowed", "retained"]), default="exact")
@click.option("--horizon", default=None, type=int)
@click.option("--max-users", default=None, type=click.IntRange(min=1))
@click.option("--save-recommendations", type=click.Path(), default=None)
def main(maxlen, embedding_dim, num_heads, dropout_rate, num_blocks, dataname,
         path, tau, k, prior, order_seed, policy, horizon, max_users,
         save_recommendations):
    """Run and print one RosE operating point as one terminal line."""
    model, inputs, answers, bprob, bi, bundlenum, itemnum = load_experiment(
        dataname, path, maxlen, embedding_dim, num_heads, dropout_rate, num_blocks)
    inputs, answers, order = reorder_users(
        inputs, answers, order_seed, max_users, source_order_indices(dataname))
    iprob = make_item_prior(prior, dataname, bi, bprob, model, k, maxlen)
    max_bundle_size = int(np.asarray(bi.sum(axis=1)).max())
    reclist = RosE(model, inputs, iprob, itemnum, max_bundle_size, bi,
                   tau=tau, k=k, horizon=horizon, policy=policy)
    recommendations = reclist.cpu().numpy()
    result = metrics("rose", dataname, k, recommendations, answers, bi,
                     bundlenum, itemnum)
    model_config = {"maxlen": maxlen, "embedding_dim": embedding_dim,
                    "num_heads": num_heads, "dropout_rate": dropout_rate,
                    "num_blocks": num_blocks}
    checkpoint = checkpoint_identity(dataname, path)
    result.update(tau=tau, prior=prior, order_seed=order_seed,
                  policy=policy, horizon=horizon, checkpoint=checkpoint,
                  model_config=model_config, case_count=len(recommendations),
                  max_users=max_users)
    save_pickle(
        save_recommendations, recommendations, answers, order,
        method="rose", dataname=dataname, k=k, tau=tau, prior=prior,
        order_seed=order_seed, policy=policy, horizon=horizon,
        item_prior=iprob, checkpoint=checkpoint, model_config=model_config,
        max_users=max_users,
    )
    emit(result)


if __name__ == "__main__":
    main()
