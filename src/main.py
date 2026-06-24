from tqdm import tqdm
import click

import torch
import numpy as np
from scipy.sparse import csr_matrix

from model import SBR
from main_util import *


def RosE(model, test_input, iprob, itemnum, max_bundle_size, BIsparse, tau=1, k=10):
    newreclist = []

    ifreq = np.zeros(itemnum+1)

    for u in tqdm(range(len(test_input))):
        pred = model.predict(torch.LongTensor(test_input[u]).to(model.device))
        pred = pred.squeeze().cpu().detach().numpy()

        bpref = np.insert(pred, 0, 0)

        expifreq = ifreq + (len(test_input)-u-1) * k * iprob

        expiprob = expifreq / sum(expifreq)
        ent0 = -np.sum(expiprob[1:] * np.log2(expiprob[1:]))
        expiprob1 = expifreq / (sum(expifreq)+1)
        ent1 = -np.sum(expiprob1[1:] * np.log2(expiprob1[1:]))

        expiprob2 = (expifreq + 1) / (sum(expifreq)+1)
        idivgain = (ent1-ent0) - (expiprob2[1:] * np.log2(expiprob2[1:])) + (expiprob1[1:] * np.log2(expiprob1[1:]))
        idivgain /= max(np.abs(idivgain))
        idivgain = np.insert(idivgain, 0, 0)

        bdivgain = BIsparse.dot(idivgain) / max_bundle_size

        divgain = bdivgain

        score = softmax(bpref[1:], temperature=tau)
        score = min_max_normalize(score)
        score = np.insert(score, 0, 0)

        g = score + (1-score)*divgain
        
        reclist = (-g).argsort()[:k]

        reclist = sorted(reclist, reverse=True, key=lambda b: bpref[b])

        for b in reclist:
            ifreq += BIsparse[b].toarray().squeeze()
        newreclist.append(reclist)

    newreclist = torch.LongTensor(newreclist)

    return newreclist


@click.command()
@click.option('--maxlen', default = 200)
@click.option('--embedding_dim', default=128)
@click.option('--num_heads', default=1)
@click.option('--dropout_rate', default=0.5)
@click.option('--num_blocks', default=2)
@click.option('--dataname', default='chess')
@click.option('--path', default=None)
@click.option('--tau', default=2.1)
@click.option('--k', default=10)
def main(
        maxlen,
        embedding_dim,
        num_heads,
        dropout_rate,
        num_blocks,
        dataname,
        path,
        tau,
        k
    ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    b_i_file = open(f'../data/{dataname}/bundle-item.txt', 'r')

    print('Read data...')
    item_num=0
    bundle_item_list_dict= defaultdict(list)
    for line in b_i_file:
        bundle_id, item_id = line.rstrip().split('\t')
        bundle_id = int(bundle_id)
        item_id = int(item_id)
        item_num = max(item_id, item_num)
        bundle_item_list_dict[bundle_id].append(item_id)

    max_bundle_size = max(len(lst) for lst in bundle_item_list_dict.values())

    print('Preprocess data...')
    its = {}
    for b in bundle_item_list_dict:
        for i in bundle_item_list_dict[b]:
            if i not in its:
                its[i] = True

    itemnum = len(its)

    data = []
    row_indices = []
    col_indices = []
    for b in bundle_item_list_dict:
        for i in bundle_item_list_dict[b]:
            data.append(1.0)
            row_indices.append(b)
            col_indices.append(i)

    BIsparse = csr_matrix((data, (row_indices, col_indices)))

    bundle_item_vector_dict = {}
    for b in bundle_item_list_dict:
        ifreq = np.zeros(itemnum+1)
        for i in bundle_item_list_dict[b]:
            ifreq[i] += 1
        bundle_item_vector_dict[b] = ifreq

    usernum, bundlenum, test_input, test_ans, bprob = test_data(dataname, maxlen)

    iprob = np.zeros(itemnum+1)
    for b in range(1, bundlenum+1):
        iprob += BIsparse[b].toarray().squeeze() * bprob[b]

    print('Load the model...')
    model = SBR(bundlenum, item_num, bundle_item_list_dict, max_bundle_size, device,
            maxlen=maxlen, embedding_dim=embedding_dim, num_heads=num_heads, 
            dropout_rate=dropout_rate, num_blocks=num_blocks).to(device)

    if not path:
        model_path = f'../data/{dataname}/pretrained.pt'
    else:
        model_path = path
    model.load_state_dict(torch.load(model_path, map_location = device))

    model.eval()

    print('Generate recommendations...')
    reclist = RosE(model, test_input, iprob, itemnum, max_bundle_size, BIsparse, tau=tau, k=k)

    ndcg, ent, gini = eval(reclist, test_ans, BIsparse, bundlenum, itemnum, k=k)
    print()
    print(f'nDCG@{k} : {ndcg:.3f}')
    print(f'ENT@{k}  : {ent:.3f}')
    print(f'Gini@{k} : {gini:.3f}')
    print()

if __name__ == '__main__':
    main()