import sys 
sys.path.append('.')
sys.path.append('..')

import torch 
from torch import nn 
from torch.nn import functional as F
from llm.layers.transformer import TransformerBlock


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, pad_idx=0, max_length=1024, num_blocks=6):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.pos_encoding = nn.Embedding(max_length, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, padding_mask=None):
        # Input embeddings
        x = self.embeddings(input_ids)
        x = x + self.pos_encoding(torch.arange(input_ids.size(1), device=input_ids.device))

        # Transformer blocks 
        padding_mask = padding_mask == 0
        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)
        
        # Output probabilities
        logits = self.output_layer(x)

        return logits
    
if __name__=="__main__":
    model = Model(100, 256, 4, 0)

    input_ids = torch.randint(1, 100, (32, 25))
    padding_mask = torch.randint(0, 2, (32, 25))

    logits = model(input_ids, padding_mask)
    print(logits.shape)