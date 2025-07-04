import torch 
from torch import nn 
from torch.nn import functional as F


# DyT Module to substitute LayerNormalization [Zhu et al. 2025]
class DyT(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        x =  self.gamma * x + self.beta
        return x 
    

class SilMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y 
    

# Transformer Block for GPT 
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
            )
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            SilMul(),
            nn.Linear(embed_dim // 2, embed_dim),
            SilMul(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        # Mask for autoregressive learning
        BSZ, SEQ_LEN, EMBED_DIM = x.shape
        M = torch.full((SEQ_LEN, SEQ_LEN), float('-inf'), device=x.device)
        M = torch.triu(M, diagonal=1)

        # Multi head attention, layer norm and feedforward network
        mha_output, _ = self.multi_head_attention(x, x, x, attn_mask=M, key_padding_mask=key_padding_mask)
        x = self.layer_norm_1(x + mha_output)

        ffn_output = self.feed_forward(x)
        x = self.layer_norm_2(x + ffn_output)

        return x


if __name__=="__main__":
    model = TransformerBlock(256, 4)

    x = torch.randn((32, 5, 256))
    y = model(x)
    print(y.shape)