# import torch
# from torch import nn
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, d_k):
#         super().__init__()
#         self.d_k = d_k
#
#     def forward(self, query, key, value, coords,mask=None,temperature=1):
#         coords_diff=coords.unsqueeze(1)-coords.unsqueeze(2)
#         coords_norm = 1-torch.norm(coords_diff, dim=-1).unsqueeze(1)
#         epsilon = 1e-10
#         # Sum along the row (dim=-1 for the last dimension, which is the length dimension in this case)
#         row_sums = coords_norm.sum(dim=-1, keepdim=True) + epsilon
#         # Normalize each row to sum to 1
#         coords_norm_l1 = coords_norm / row_sums
#         #print(coords_norm)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)/temperature
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#             #print(scores.shape)
#             scores = scores.masked_fill(mask == 1, float('-inf'))  # Masking with -inf before softmax
#         #print(coords_norm_l1)
#         attention = F.softmax(scores, dim=-1)+coords_norm_l1
#         output = torch.matmul(attention, value)
#         return output, attention
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, nhead):
#         super().__init__()
#         assert d_model % nhead == 0, "d_model must be divisible by nhead"
#         self.d_k = d_model // nhead
#         self.nhead = nhead
#         self.linear_q = nn.Linear(d_model, d_model)
#         self.linear_k = nn.Linear(d_model, d_model)
#         self.linear_v = nn.Linear(d_model, d_model)
#         self.attention = ScaledDotProductAttention(self.d_k)
#         self.out_proj = nn.Linear(d_model, d_model)
#
#     def forward(self, query, key, value, coords,mask=None):
#         batch_size = query.size(0)
#         query, key, value = [linear(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
#                              for linear, x in zip((self.linear_q, self.linear_k, self.linear_v), (query, key, value))]
#         x, _ = self.attention(query, key, value, coords,mask)
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
#         return self.out_proj(x)
#
# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super().__init__()
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)
#
#     def forward(self, x):
#         return self.linear2(F.relu(self.linear1(x)))
#
# class SublayerConnection(nn.Module):
#     def __init__(self, d_model, dropout):
#         super().__init__()
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, sublayer):
#         return x + self.dropout(sublayer(self.norm(x)))
#
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, d_ff, dropout):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, nhead)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
#         self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, src, coords,src_mask=None):
#         src = self.sublayer[0](src, lambda x: self.self_attn(x, x, x, coords,src_mask))
#         src = self.sublayer[1](src, self.feed_forward)
#         return self.norm(src)
#
# class TransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
#         self.num_layers = num_layers
#
#     def forward(self, src, coords,mask=None):
#         output = src
#         for layer in self.layers:
#             output = layer(output, coords,mask)
#         return output
#
# class CrystalTransformer(nn.Module):
#     def __init__(self, feature_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
#         super(CrystalTransformer, self).__init__()
#         self.positional_encoding = nn.Parameter(torch.rand(1, 1024, feature_size))
#         self.coords_embed=nn.Linear(3,feature_size//2)
#         self.atom_embed=nn.Linear(100,feature_size//2)
#         encoder_layers = TransformerEncoderLayer(d_model=feature_size,
#                                                  nhead=num_heads,
#                                                  d_ff=dim_feedforward,
#                                                  dropout=dropout)
#
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
#         self.output_linear = nn.Linear(feature_size, 1)  # Assuming regression task
#
#     def forward(self, atom,coords,mask):
#         atom_=self.atom_embed(atom)
#         coords_=self.coords_embed(coords)
#
#         src=torch.cat([atom_,coords_],dim=-1)
#         batch_max_len = src.size(1)
#         position_encoding = self.positional_encoding[:, :batch_max_len, :]
#         # print(position_encoding.shape)
#         # print(src.shape)
#         #src += position_encoding  # Adding positional encoding
#         mask_=torch.bmm(mask.unsqueeze(-1).float(),mask.unsqueeze(1).float())
#         #print(mask_.shape,mask_)
#         output = self.transformer_encoder(src,coords,mask=mask_)
#         output = self.output_linear(src[:, 0, :])  # Getting the output for the [CLS] token or similar
#         return output

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CrystalTransformer(nn.Module):
    def __init__(self, feature_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super(CrystalTransformer, self).__init__()
        self.positional_encoding = nn.Parameter(torch.rand(1, 256, feature_size))
        self.coords_embed=nn.Linear(3,feature_size//2)
        self.atom_embed=nn.Linear(100,feature_size//2)
        self.coord_diff_embed = nn.Linear(3, feature_size // 4)
        encoder_layers = TransformerEncoderLayer(d_model=feature_size,
                                                 nhead=num_heads,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_linear1 = nn.Linear(feature_size, 128)  # Assuming regression task
        self.output_linear2 = nn.Linear(128, 1)
    def forward(self, atom,coords,mask):
        atom=self.atom_embed(atom)
        coords=self.coords_embed(coords)
        mask_diff=(~mask).float().unsqueeze(-1)*(~mask).float().unsqueeze(-2)
        src = torch.cat([atom, coords], dim=-1)
        batch_max_len = src.size(1)
        #position_encoding = self.positional_encoding[:, :batch_max_len, :]
        #src += position_encoding  # Adding positional encoding
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = self.output_linear1(output[:, 0, :])
        output = self.output_linear2(output)# Getting the output for the [CLS] token or similar
        return output

    def compute_coords_diff_feature(self,coords,mask):
        coords_diff = coords.unsqueeze(1) - coords.unsqueeze(2)
        coords_diff_norm = torch.norm(coords_diff, dim=-1)
        coords_diff_topk_idx = torch.topk(coords_diff_norm, k=5, dim=-1)[1]
        coords_diff_gather = coords_diff.gather(1, coords_diff_topk_idx)
