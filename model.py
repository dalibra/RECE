from math import ceil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecBackBone(nn.Module):
    def __init__(self, item_num, config):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, config['hidden_units'], padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)

        for _ in range(config['num_blocks']):
            new_attn_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                config['hidden_units'],config['num_heads'],config['dropout_rate']
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_units'], config['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(config['manual_seed'])
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        log_feats = log_feats.view(-1, log_feats.shape[-1])
        pos_seqs = pos_seqs.view(-1)
        neg_seqs =  neg_seqs.permute(0, 2, 1).reshape(-1, neg_seqs.shape[1]) # (bs * seq, neg)

        pos_embs = self.item_emb(pos_seqs) # (bs * seq, hd)
        neg_embs = self.item_emb(neg_seqs) # (bs * seq, neg, hd)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :] * neg_embs).sum(dim=-1) # (bs * seq, neg)

        return pos_seqs, pos_logits, neg_logits

    def score(self, seq):
        '''
        Takes 1d sequence as input and returns prediction scores.
        '''
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRec(SASRecBackBone):
    def __init__(self, item_num, config):
        super().__init__(item_num + 1, config)

        self.fwd_type = config['fwd_type']

        if self.fwd_type in ['bce', 'gbce']:
            self.n_neg_samples = config['n_neg_samples']

        if self.fwd_type == 'rece':
            self.n_buckets = config['n_buckets']
            self.buckets_per_chunk = config['buckets_per_chunk']
            self.n_extra_chunks = config['n_extra_chunks']
            self.rounds = config['rounds']

            padded_ds = ceil(self.item_emb.weight.shape[0] / self.n_buckets) * self.n_buckets

            self.item_emb = nn.Embedding(padded_ds, config['hidden_units'], padding_idx=self.pad_token)
            torch.nn.init.xavier_uniform_(self.item_emb.weight.data)

            with torch.no_grad():
                self.item_emb.weight[self.pad_token+1:, :] = 0.
            
        elif self.fwd_type == 'gbce':
            alpha = self.n_neg_samples / (item_num - 1.)
            self.beta = alpha * (config['gbce_t'] * (1. - 1. / alpha) + 1. / alpha)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        if self.fwd_type == 'rece':
            return self.rece_forward(log_seqs, pos_seqs)

        elif self.fwd_type == 'bce':
            return self.bce_forward(log_seqs, pos_seqs, neg_seqs)
        
        elif self.fwd_type == 'gbce':
            return self.gbce_forward(log_seqs, pos_seqs, neg_seqs)

        elif self.fwd_type == 'ce':
            return self.ce_forward(log_seqs, pos_seqs)
        
        elif self.fwd_type == 'dross':
            return self.dross_forward(log_seqs, pos_seqs, neg_seqs)

        else:
            raise ValueError(f'Wrong fwd_type type - {self.fwd_type}')

    def bce_forward(self, log_seqs, pos_seqs, neg_seqs):
        device = log_seqs.device
        pos_seqs, pos_logits, neg_logits = super().forward(log_seqs, pos_seqs, neg_seqs)

        pos_logits = pos_logits[:, None]

        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)

        logits = torch.cat([pos_logits, neg_logits], -1)

        gt = torch.cat([pos_labels, neg_labels], -1)

        mask = (pos_seqs != self.pad_token).float()

        loss_per_element = \
            torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1) * mask
        loss = loss_per_element.sum() / mask.sum()
        
        return loss

    def gbce_forward(self, log_seqs, pos_seqs, neg_seqs):
        device = log_seqs.device

        pos_seqs, pos_logits, neg_logits = super().forward(log_seqs, pos_seqs, neg_seqs)

        pos_logits = pos_logits[:, None]

        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)

        pos_logits = torch.log(1 / (F.sigmoid(pos_logits) ** (- self.beta) - 1.))

        logits = torch.cat([pos_logits, neg_logits], -1)

        gt = torch.cat([pos_labels, neg_labels], -1)

        mask = (pos_seqs != self.pad_token).float()

        loss_per_element = \
            torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1) * mask
        loss = loss_per_element.sum() / mask.sum()

        return loss
    
    def dross_forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs) # bs * seq, hd

        pos_embs = self.item_emb(pos_seqs) # bs * seq, hd
        neg_embs = self.item_emb(neg_seqs) # bs, n_neg, hd

        pos_logits = (log_feats * pos_embs).sum(dim=-1)[:, :, None].view(-1, 1) # bs * seq, 1
        neg_logits = \
            (log_feats[:, :, None, :] * neg_embs[:, None, :, :]).sum(dim=-1).view(-1, neg_embs.shape[-2]) # bs * seq, n_neg

        logits = torch.cat([pos_logits, neg_logits], dim=-1) # bs * seq, 1 + n_neg
        labels = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)

        mask = (pos_seqs != self.pad_token).float()

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='none') * mask.view(-1)
        loss = loss.sum() / mask.sum()

        return loss

    def ce_forward(self, log_seqs, pos_seqs):
        emb = self.log2feats(log_seqs)

        logits = emb @ self.item_emb.weight.T
        indices = torch.where(pos_seqs.view(-1) != self.pad_token)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos_seqs.view(-1)[indices], reduction='mean')
        return loss
    
    def rece_forward(self, log_seqs, pos_seqs):
        emb = self.log2feats(log_seqs)
        hd = emb.shape[-1]

        mask = log_seqs.view(-1) != self.pad_token
        x = emb.view(-1, emb.shape[-1])
        x = x[mask, :]
        bs = x.shape[0]

        w = self.item_emb.weight
        ds = w.shape[0]

        n_chunks = self.n_buckets // self.buckets_per_chunk

        padded_bs = ceil(x.shape[0] / self.n_buckets) * self.n_buckets

        x = F.pad(x, (0, 0, 0, padded_bs - bs), 'constant', 0)
        y = pos_seqs.view(-1)
        y = y[mask]
        y = F.pad(y, (0, padded_bs - bs), 'constant', self.pad_token)

        bs = x.shape[0]
        ds = w.shape[0]

        chunk_size_x = x.shape[0] // n_chunks
        chunk_size_y = w.shape[0] // n_chunks

        catalog = torch.clip(torch.arange(ds, device=x.device, dtype=torch.int32), min=0, max=self.pad_token)

        true_class_logits = (x * torch.index_select(w, dim=0, index=y)).sum(dim=1)[:, None] # (bs, 1)

        buckets = torch.randn(self.rounds, self.n_buckets, hd, device=x.device) # (rounds, n_buckets, hd)

        with torch.no_grad():
            x_bucket = buckets @ x.T # (rounds, n_b, hd) x (hd, bs) -> (rounds, n_buckets, bs)
            x_ind = torch.argsort(torch.argmax(x_bucket, dim=1)) # (rounds, bs)
            del x_bucket

            y_bucket = buckets @ w.T # (rounds, n_b, hd) x (hd, ds) -> (rounds, n_buckets, ds)
            y_ind = torch.argsort(torch.argmax(y_bucket, dim=1)) # (rounds, ds)
            del y_bucket, buckets

            catalog = torch.take_along_dim(catalog, y_ind.view(-1), 0) \
                .view(self.rounds, n_chunks, chunk_size_y) # is needed for accounting for duplicates when rounds > 1
            catalog = F.pad(catalog, 
                            (0, 0, self.n_extra_chunks, self.n_extra_chunks),
                            'constant', self.pad_token) # (rounds, n_chunks+n_extra_chunks*2, chunk_size_y)
            catalog = catalog.unfold(1, n_chunks, 1) \
                             .permute(0, 3, 1, 2) \
                             .view(self.rounds, n_chunks, -1) # (rounds, n_chunks, (1+2*n_extra_chunks) * chunk_size_y)
            catalog_ = \
                catalog[:, :, None, :] \
                    .expand(-1, -1, chunk_size_x, -1) \
                    .reshape(catalog.shape[0], -1, catalog.shape[-1]) 
                    # (rounds, n_chunks * chunk_size_x, (1+2*n_extra_chunks) * chunk_size_y)
            catalog = torch.zeros_like(catalog_) \
                           .scatter_(1, x_ind[:, :, None] \
                           .expand_as(catalog_), catalog_) 
                           # same shape, but now ordered as originally, before it was ordered according to chunks
            catalog = catalog.permute(1, 0, 2) \
                             .reshape(catalog.shape[1], -1) 
                             # (n_chunks * chunk_size_x, rounds * (1+2*n_extra_chunks) * chunk_size_y))
            catalog_sorted = torch.sort(catalog)[0]
            catalog_counts = torch.searchsorted(catalog_sorted, catalog, side='right', out_int32=True)
            catalog_counts2 = torch.searchsorted(catalog_sorted, catalog, side='left', out_int32=True)
            del catalog_sorted
            catalog_counts -= catalog_counts2
            del catalog_counts2
            catalog_counts = catalog_counts.float().log_()
            catalog_mask = (catalog == self.pad_token) | (catalog == y[:, None])

            # mask pad token logits and positive class logits
            catalog_counts = catalog_counts.masked_fill(catalog_mask, float('inf'))
            del catalog, catalog_mask

        x_sorted = torch.take_along_dim(x, x_ind.view(-1, 1), 0) \
                        .view(self.rounds, n_chunks, chunk_size_x, -1) # (rounds, n_chunks, chunk_size_x, hd)

        y_sorted = torch.take_along_dim(w, y_ind.view(-1, 1), 0) \
                        .view(self.rounds, n_chunks, chunk_size_y, -1) # (rounds, n_chunks, chunk_size_y, hd)
        y_sorted = F.pad(y_sorted,
                         (0, 0, 0, 0, self.n_extra_chunks, self.n_extra_chunks),
                         'constant', 0)
                          # (rounds, n_chunks+n_extra_chunks*2, chunk_size_y, hd),
                          # so that the first and the last chunks could look backward and forward
        y_sorted = y_sorted.unfold(1, n_chunks, 1) \
                            .permute(0, 4, 1, 2, 3) \
                            .view(self.rounds, n_chunks, -1, hd) \
                            .permute(0, 1, 3, 2) 
                            # (rounds, n_chunks, hd, (1+2*n_extra_chunks) * chunk_size_y)
                            # adding previous and later chunks

        # (rounds, n_chunks, chunk_size_x, (1+2*n_extra_chunks) * chunk_size_y) (x with cur_y, prev_ys, next_ys)
        wrong_class_logits_ = x_sorted @ y_sorted
        wrong_class_logits_ = wrong_class_logits_ \
            .view(self.rounds, -1, wrong_class_logits_.shape[-1]) 
            # (rounds, n_chunks * chunk_size_x, (1+2*n_extra_chunks) * chunk_size_y)
        wrong_class_logits = torch.zeros_like(wrong_class_logits_) \
                                  .scatter_(1, x_ind[:, :, None].expand_as(wrong_class_logits_), wrong_class_logits_) 
                                  # same shape, but now ordered as originally, before it was ordered according to chunks
        del wrong_class_logits_
        wrong_class_logits = wrong_class_logits.permute(1, 0, 2)\
                                               .reshape(wrong_class_logits.shape[1], -1) 
                                               # (n_chunks * chunk_size_x, rounds * (1+2*n_extra_chunks) * chunk_size_y))

        # aсcount for duplicates
        wrong_class_logits -= catalog_counts # (n_chunks * chunk_size_x, rounds * (1+2*n_extra_chunks) * chunk_size_y))

        logits = torch.cat((wrong_class_logits, true_class_logits), dim=1)

        indices = torch.where(y != self.pad_token)

        loss = F.cross_entropy(logits[indices],
                               (logits.shape[-1] - 1) \
                               * torch.ones(logits.shape[0],
                                            dtype=torch.int64,
                                            device=logits.device)[indices]
        )

        return loss
