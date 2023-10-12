import torch.nn as nn

import torch as th
import torch_geometric.nn as pygnn
from torch_geometric.utils import to_dense_batch
import pandas as pd



class Atten_transformer(th.nn.Module):
    def __init__(self,out_feats):
        '''

        :param in_feats: 5000
        :param hid_feats: 64
        :param out_feats: 64
        '''
        super(Atten_transformer, self).__init__()
        self.heads = 4
        self.attn_dropout = 0.0
        self.dropout = 0.0
        self.log_attn_weights = False
        self.layer_norm = False
        self.batch_norm = True
        self.isNeedAttn = False
        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")


        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(out_feats)
            self.norm1_attn = pygnn.norm.LayerNorm(out_feats)

        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(out_feats)
            self.norm1_attn = nn.BatchNorm1d(out_feats)
        self.dropout_local = nn.Dropout(self.dropout)
        self.dropout_attn = nn.Dropout(self.dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(out_feats, out_feats)
        self.ff_linear2 = nn.Linear(out_feats, out_feats)
        self.act_fn_ff = nn.ReLU()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(out_feats)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(out_feats)
        self.ff_dropout1 = nn.Dropout(self.dropout)
        self.ff_dropout2 = nn.Dropout(self.dropout)
        self.self_attn = th.nn.MultiheadAttention(out_feats, self.heads, dropout=self.attn_dropout, batch_first=True)

    def forward(self, x, batchindex, attn_mask=None):
        h_in1 = x
        h_out_list = []
        h_out_list.append(x)
        h_dense, key_padding_mask = to_dense_batch(x, batchindex)
        key_padding_mask = ~key_padding_mask

        if self.isNeedAttn:
            x, A = self.self_attn(h_dense, h_dense, h_dense,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=True)
            self.attn_weights = A.detach().cpu()


            target_select_node_id = 2

            key_padding_mask_0 = key_padding_mask[target_select_node_id].cpu()
            attn_mask_0 = self.attn_weights[target_select_node_id]

            valid_indices = th.nonzero(~key_padding_mask_0).squeeze(-1)



            filtered_attn = attn_mask_0.index_select(0, valid_indices).index_select(1, valid_indices)
            df = pd.DataFrame(filtered_attn.numpy())


            file_path = "attention_A/A.csv"
            df.to_csv(file_path, index=False, header=False)

        else:
            x = self.self_attn(h_dense, h_dense, h_dense,

                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]

        h_attn = x[~key_padding_mask]

        h_attn = self.dropout_attn(h_attn)

        h_attn = h_in1 + h_attn
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batchindex)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)
        h = sum(h_out_list)
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batchindex)
        if self.batch_norm:
            h = self.norm2(h)

        return h

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))