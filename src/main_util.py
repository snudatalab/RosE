import numpy as np

from scipy.special import entr
from collections import defaultdict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, temperature=1):
    exp_x = np.exp((x - np.max(x))/temperature)
    return exp_x / np.sum(exp_x)

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def eval(reclist, ans, BIsparse, bund_num, item_num, k=None):
    if k is None:
        k = len(reclist[0])
    
    ndcg = 0
    for i in range(len(reclist)):
        answer = ans[i]
        if answer in reclist[i, :k]:
            indices = np.where(reclist[i, :k] == answer)[0]
            rank = indices[0]
            ndcg += 1 / np.log2(rank + 2)
    ndcg /= len(reclist)
    
    bcnt = np.zeros(bund_num+1)
    for rl in reclist:
        for b in rl:
            bcnt[b] += 1
    
    itemcnt = np.zeros(item_num+1)
    for b in range(1, bund_num+1):
        for i in BIsparse[b].indices:
            itemcnt[i] += bcnt[b]
    
    ent = entr(itemcnt / itemcnt.sum()).sum()
    
    return ndcg, ent

def test_data(fname, maxlen=200):
    usernum = 0
    bundlenum = 0
    UB = defaultdict(list)
    UT = defaultdict(list)
    bcnt = defaultdict(int)

    test_time = defaultdict(int)
    # assume user/item index starting from 1
    
    f = open('../data/'+fname+'/user-bundle.txt', 'r')
    for line in f:
        u, b, t = line.rstrip().split('\t')
        u = int(u)
        b = int(b)
        t = int(t)
        bcnt[b] += 1
        usernum = max(u, usernum)
        bundlenum = max(b, bundlenum)
        UB[u].append(b)
        UT[u].append(t)
        test_time[u] = max(test_time[u], t)

    us = list(range(1, usernum+1))
    us = sorted(us, key=lambda x: test_time[x])

    test_input = [UB[u][:-1] for u in us]
    test_ans = [UB[u][-1] for u in us]

    for u in range(len(test_input)):
        seq = np.zeros(maxlen)
        for i in range(1, min(maxlen+1, len(test_input[u])+1)):
            seq[-i] = test_input[u][-i]
        test_input[u] = seq
    
    bprob = np.zeros(bundlenum+1)
    for b in range(1, bundlenum+1):
        bprob[b] = bcnt[b]
    bprob /= sum(bprob)

    return usernum, bundlenum, test_input, test_ans, bprob