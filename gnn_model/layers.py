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

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:  # 默认是False
            self.norm1_local = pygnn.norm.LayerNorm(out_feats)
            self.norm1_attn = pygnn.norm.LayerNorm(out_feats)

        if self.batch_norm:  # 这个会跑
            self.norm1_local = nn.BatchNorm1d(out_feats)
            self.norm1_attn = nn.BatchNorm1d(out_feats)
        self.dropout_local = nn.Dropout(self.dropout)
        self.dropout_attn = nn.Dropout(self.dropout)

        # Feed Forward block. 下面定义前馈层。
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
            # weight_attn = A[~key_padding_mask]

            target_select_node_id = 2
            # for target_select_node_id in range(0, 31):
            key_padding_mask_0 = key_padding_mask[target_select_node_id].cpu()
            attn_mask_0 = self.attn_weights[target_select_node_id]
            # 获取key_padding_mask中为False的索引
            valid_indices = th.nonzero(~key_padding_mask_0).squeeze(-1)
            # shape0 = valid_indices.shape[0]
            # if valid_indices.shape[0] == 141:
            # print("target_select_node_id为：{}".format(target_select_node_id))

            # 使用这些索引从attn_mask中选择相应的行和列
            filtered_attn = attn_mask_0.index_select(0, valid_indices).index_select(1, valid_indices)
            df = pd.DataFrame(filtered_attn.numpy())

            # 保存到CSV文件
            file_path = "attention_A/A.csv"
            df.to_csv(file_path, index=False, header=False)

        else:
            x = self.self_attn(h_dense, h_dense, h_dense,
                                ##self.self_attn在前面实例化类的时候实例过了，为：self.self_attn = torch.nn.MultiheadAttention( ##实例化多头注意力类dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]

        h_attn = x[~key_padding_mask]

        h_attn = self.dropout_attn(h_attn)  ##在上面返回的h_attn,其维度已经变回了{Tensor:(640,64)}

        h_attn = h_in1 + h_attn  # Residual connection.和输出的x进行残差连接。
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batchindex)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)  ##把进过残差后的节点表示特征加入到h_out_list列表中。
        h = sum(h_out_list)
        h = h + self._ff_block(h)  # 经过一个前馈层，维度不变
        if self.layer_norm:
            h = self.norm2(h, batchindex)
        if self.batch_norm:
            h = self.norm2(h)  # 经过一个批正则化，维度不变

        return h

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))  ##依次嵌套了一个linear层、激活层、dropout层。
        return self.ff_dropout2(self.ff_linear2(x))








# class Atten_transformer(th.nn.Module):
#     def __init__(self,out_feats):
#         '''
#
#         :param in_feats: 5000
#         :param hid_feats: 64
#         :param out_feats: 64
#         '''
#         super(Atten_transformer, self).__init__()
#         self.heads = 1
#         self.attn_dropout = 0.0
#         self.dropout = 0.0
#         self.log_attn_weights = False
#         self.layer_norm = False
#         self.batch_norm = True
#         if self.layer_norm and self.batch_norm:
#             raise ValueError("Cannot apply two types of normalization together")
#
#         # Normalization for MPNN and Self-Attention representations.
#         if self.layer_norm:  # 默认是False
#             self.norm1_local = pygnn.norm.LayerNorm(out_feats)
#             self.norm1_attn = pygnn.norm.LayerNorm(out_feats)
#
#         if self.batch_norm:  # 这个会跑
#             self.norm1_local = nn.BatchNorm1d(out_feats)
#             self.norm1_attn = nn.BatchNorm1d(out_feats)
#         self.dropout_local = nn.Dropout(self.dropout)
#         self.dropout_attn = nn.Dropout(self.dropout)
#
#         # Feed Forward block. 下面定义前馈层。
#         self.ff_linear1 = nn.Linear(out_feats, out_feats)
#         self.ff_linear2 = nn.Linear(out_feats, out_feats)
#         self.act_fn_ff = nn.ReLU()
#         if self.layer_norm:
#             self.norm2 = pygnn.norm.LayerNorm(out_feats)
#         if self.batch_norm:
#             self.norm2 = nn.BatchNorm1d(out_feats)
#         self.ff_dropout1 = nn.Dropout(self.dropout)
#         self.ff_dropout2 = nn.Dropout(self.dropout)
#         self.self_attn_1 = th.nn.MultiheadAttention(out_feats, self.heads, dropout=self.attn_dropout,
#                                                    batch_first=True)
#         self.self_attn_2 = th.nn.MultiheadAttention(out_feats, self.heads, dropout=self.attn_dropout,
#                                                     batch_first=True)
#         self.self_attn_3 = th.nn.MultiheadAttention(out_feats, self.heads, dropout=self.attn_dropout,
#                                                     batch_first=True)
#         self.self_attn_4 = th.nn.MultiheadAttention(out_feats, self.heads, dropout=self.attn_dropout,
#                                                     batch_first=True)
#
#     def forward(self, x, batchindex, head_list=None):
#         h_in1 = x
#         h_out_list = []
#         h_attn_outputs = []
#         h_out_list.append(x)
#         h_dense, key_padding_mask = to_dense_batch(x, batchindex)
#         key_padding_mask = ~key_padding_mask
#
#         x_1 = self.self_attn_1(h_dense, h_dense, h_dense,
#                             ##self.self_attn在前面实例化类的时候实例过了，为：self.self_attn = torch.nn.MultiheadAttention( ##实例化多头注意力类dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
#                             attn_mask=head_list[0],
#                             key_padding_mask=key_padding_mask,
#                             need_weights=False)[0]
#         h_attn_1 = x_1[~key_padding_mask]
#         h_attn_outputs.append(h_attn_1)
#
#         x_2 = self.self_attn_2(h_dense, h_dense, h_dense,
#                             ##self.self_attn在前面实例化类的时候实例过了，为：self.self_attn = torch.nn.MultiheadAttention( ##实例化多头注意力类dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
#                             attn_mask=head_list[1],
#                             key_padding_mask=key_padding_mask,
#                             need_weights=False)[0]
#         h_attn_2 = x_2[~key_padding_mask]
#         h_attn_outputs.append(h_attn_2)
#
#         x_3 = self.self_attn_3(h_dense, h_dense, h_dense,
#                             ##self.self_attn在前面实例化类的时候实例过了，为：self.self_attn = torch.nn.MultiheadAttention( ##实例化多头注意力类dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
#                             attn_mask=head_list[2],
#                             key_padding_mask=key_padding_mask,
#                             need_weights=False)[0]
#         h_attn_3 = x_3[~key_padding_mask]
#         h_attn_outputs.append(h_attn_3)
#
#         x_4 = self.self_attn_4(h_dense, h_dense, h_dense,
#                             ##self.self_attn在前面实例化类的时候实例过了，为：self.self_attn = torch.nn.MultiheadAttention( ##实例化多头注意力类dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
#                             attn_mask=head_list[3],
#                             key_padding_mask=key_padding_mask,
#                             need_weights=False)[0]
#         h_attn_4 = x_4[~key_padding_mask]
#         h_attn_outputs.append(h_attn_4)
#
#         h_attn = sum(h_attn_outputs)/4 #好奇怪，为啥不是0,而是1
#
#         h_attn = self.dropout_attn(h_attn)  ##在上面返回的h_attn,其维度已经变回了{Tensor:(640,64)}
#
#         h_attn = h_in1 + h_attn  # Residual connection.和输出的x进行残差连接。
#         if self.layer_norm:
#             h_attn = self.norm1_attn(h_attn, batchindex)
#         if self.batch_norm:
#             h_attn = self.norm1_attn(h_attn)
#         h_out_list.append(h_attn)  ##把进过残差后的节点表示特征加入到h_out_list列表中。
#         h = sum(h_out_list)
#         h = h + self._ff_block(h)  # 经过一个前馈层，维度不变
#         if self.layer_norm:
#             h = self.norm2(h, batchindex)
#         if self.batch_norm:
#             h = self.norm2(h)  # 经过一个批正则化，维度不变
#
#         return h
#
#     def _ff_block(self, x):
#         """Feed Forward block.
#         """
#         x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))  ##依次嵌套了一个linear层、激活层、dropout层。
#         return self.ff_dropout2(self.ff_linear2(x))






