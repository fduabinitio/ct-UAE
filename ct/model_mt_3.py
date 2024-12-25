
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
        self.output_linear1 = nn.Sequential(nn.Linear(feature_size, feature_size),nn.ReLU(),nn.Linear(feature_size,1))  # Assuming regression task
        self.output_linear2 = nn.Sequential(nn.Linear(feature_size, feature_size),nn.ReLU(),nn.Linear(feature_size,1))
        self.output_linear3 = nn.Sequential(nn.Linear(feature_size, feature_size),nn.ReLU(),nn.Linear(feature_size,1))
    def forward(self, atom,coords,mask):
        atom=self.atom_embed(atom)
        coords=self.coords_embed(coords)
        mask_diff=(~mask).float().unsqueeze(-1)*(~mask).float().unsqueeze(-2)
        src = torch.cat([atom, coords], dim=-1)
        batch_max_len = src.size(1)
        #position_encoding = self.positional_encoding[:, :batch_max_len, :]
        #src += position_encoding  # Adding positional encoding
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output1 = self.output_linear1(output[:, 0, :])
        output2 = self.output_linear2(output[:, 0, :])
        output3 = self.output_linear3(output[:, 0, :])# Getting the output for the [CLS] token or similar
        return (output1,output2,output3)

