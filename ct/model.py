
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
