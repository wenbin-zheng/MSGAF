import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(39, embed_size, bias=False)
        self.keys = nn.Linear(39, embed_size, bias=False)
        self.queries = nn.Linear(39, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, sequence_length, _ = x.shape

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Attention mechanism here, simplified for explanation
        # Normally includes scaling and mask for batch and sequence
        attention = torch.softmax(
            torch.matmul(queries, keys.transpose(-1, -2)) / self.embed_size ** 0.5, dim=-1
        )

        out = torch.matmul(attention, values)
        out = out.reshape(N, sequence_length, self.embed_size)
        return self.fc_out(out)