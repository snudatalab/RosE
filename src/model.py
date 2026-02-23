import torch
import torch.nn as nn

from custom_transformer import MultiheadAttention as MyMultiheadAttention

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    
class BundleEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(BundleEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.WQ = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, item_embs, query, mask):
        proj_query = self.WQ(query)
        attn_logit = (item_embs * proj_query.unsqueeze(-2)).sum(dim=-1)
        attn_logit = attn_logit.masked_fill(mask == 0, float('-inf'))
        attn_score = torch.softmax(attn_logit, dim=-1).unsqueeze(-1)

        bund_embs = (item_embs*attn_score).sum(dim=-2)

        return bund_embs
    

class SBR(nn.Module):
    def __init__(self, bund_num, item_num, bundle_item_list_dict, max_bundle_size, device,
                 maxlen = 200, embedding_dim=128, num_heads=4, dropout_rate=0.2, num_blocks=4):
        super(SBR, self).__init__()

        self.device = device
        self.bund_num = bund_num
        self.item_num = item_num
        self.maxlen = maxlen

        self.embedding_dim = embedding_dim
        self.item_embedding = torch.nn.Embedding(item_num+1, self.embedding_dim, padding_idx=0)
        self.bund_embedding = torch.nn.Embedding(bund_num+1, self.embedding_dim, padding_idx=0)
        self.global_query = nn.Parameter(torch.randn(self.embedding_dim))

        self.bundle_encoder = BundleEncoder(self.embedding_dim)
        
        bundle_item_tensor = torch.zeros((bund_num+1, max_bundle_size), dtype=torch.long)

        for b in range(1, len(bundle_item_list_dict)+1):
            for i in range(len(bundle_item_list_dict[b])):
                bundle_item_tensor[b][i] = bundle_item_list_dict[b][i]
        self.bundle_item_tensor = nn.Embedding.from_pretrained(bundle_item_tensor, freeze=True)

        self.pos_emb = torch.nn.Embedding(maxlen, 2*embedding_dim)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(2*embedding_dim, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(2*embedding_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  MyMultiheadAttention(2*embedding_dim,
                                                            num_heads,
                                                            dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(2*embedding_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(2*embedding_dim, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def sess2embs(self, bundle_sessions):
        item_sessions = self.bundle_item_tensor(bundle_sessions.to(self.device))
        item_inps = self.item_embedding(item_sessions)

        return torch.concat((torch.mean(item_inps, dim=-2), self.bund_embedding(bundle_sessions.to(self.device))), dim=-1)

    def sess2feats(self, bundle_sessions):
        seqs = self.sess2embs(bundle_sessions)

        seqs *= self.item_embedding.embedding_dim ** 0.5
        shape = bundle_sessions.shape
        poss = torch.arange(shape[-1], dtype=torch.long).expand(*shape[:-1], shape[-1]).to(self.device)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (bundle_sessions == 0).to(self.device).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask,
                                            key_padding_mask=timeline_mask)
            mha_outputs = torch.nan_to_num(mha_outputs, nan=0.0)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, bundle_sessions, pos_bundles, neg_bundles):
        log_feats = self.sess2feats(bundle_sessions)
        
        pos_embs = self.sess2embs(pos_bundles)
        neg_embs = self.sess2embs(neg_bundles)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits
    
    def predict(self, bundle_sessions, bundle_indices=None):
        log_feats = self.sess2feats(bundle_sessions.view(-1, self.maxlen))

        final_feat = log_feats[:, -1, :]

        bund_embs = self.sess2embs(torch.arange(1, self.bund_num+1, dtype=torch.long).to(self.device))

        logits = bund_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
