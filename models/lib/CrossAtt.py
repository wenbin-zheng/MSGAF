import torch
import torch.nn as nn

# 跨模特注意力机制
class CrossModalityAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossModalityAttention, self).__init__()
        self.d_model = d_model

    def forward(self, query, keys, values):
        # query: [batch_size, seq_len_q, d_model]
        # keys: [batch_size, seq_len_k, d_model]
        # values: [batch_size, seq_len_v, d_model]

        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.d_model ** 0.5)

        weights = nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(weights, values)

        return output