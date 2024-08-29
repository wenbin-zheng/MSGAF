import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        Q = self.query(x)  # (batch_size, seq_len, feature_dim)
        K = self.key(x)  # (batch_size, seq_len, feature_dim)
        V = self.value(x)  # (batch_size, seq_len, feature_dim)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (feature_dim ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_out = torch.bmm(attention_weights, V)

        return attention_out
