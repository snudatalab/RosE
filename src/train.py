import click
from tqdm import tqdm

import torch
from collections import defaultdict

from train_util import *
from model import SBR

@click.command()
@click.option('--batch_size', default=256)
@click.option('--maxlen', default=200)
@click.option('--embedding_dim', default=128)
@click.option('--num_heads', default=1)
@click.option('--dropout_rate', default=0.5)
@click.option('--num_blocks', default=2)
@click.option('--n_workers', default=1)
@click.option('--lr', default=0.001)
@click.option('--num_epochs', default=201)
@click.option('--dataname', default='chess')
@click.option('--path', default=None)
def main(
        batch_size,
        maxlen,
        embedding_dim,
        num_heads,
        dropout_rate,
        num_blocks,
        n_workers,
        lr,
        num_epochs,
        dataname,
        path
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

    print('Preprocess data...')
    dataset = data_partition(dataname)

    [user_train, user_valid, user_test, usernum, bundlenum] = dataset

    num_batch = len(user_train) // batch_size

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))


    max_bundle_size = max(len(lst) for lst in bundle_item_list_dict.values())
    print('maximum bundle size:', max_bundle_size)

    model = SBR(bundlenum, item_num, bundle_item_list_dict, max_bundle_size, device,
                maxlen=maxlen, embedding_dim=embedding_dim, num_heads=num_heads, 
                dropout_rate=dropout_rate, num_blocks=num_blocks).to(device)

    for name, param in model.named_parameters():
        try:
            if 'embedding' in name:  
                torch.nn.init.xavier_normal_(param[1:])
            elif 'bundle_item' in name: continue
            else:
                torch.nn.init.xavier_normal_(param.data)
        except:
            pass 

    model.train()

    sampler = WarpSampler(user_train, usernum, bundlenum, batch_size=batch_size, maxlen=maxlen, n_workers=n_workers)

    if not path:
        model_path = f'../data/{dataname}/pretrained.pt'
    else:
        model_path = path

    epoch_start_idx = 1

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

    print('Start training...')

    for epoch in tqdm(range(epoch_start_idx, num_epochs + 1)):
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            seq, pos, neg = sampler.next_batch()
            seq = torch.stack(seq, dim=0)
            pos = torch.stack(pos, dim=0)
            neg = torch.stack(neg, dim=0)
            
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
            adam_optimizer.zero_grad()

            loss = bce_criterion(pos_logits, pos_labels)
            loss += bce_criterion(neg_logits, neg_labels)

            loss.backward()
            adam_optimizer.step()
            
            model.train()

    torch.save(model.state_dict(), model_path)
    sampler.close()


if __name__ == '__main__':
    main()