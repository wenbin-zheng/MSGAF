import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from .CrossAtt import *
from .SelfAtt import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossModalFusion(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.cross_att = CrossModalityAttention(self.args.model_args.latent_dim)
        self.self_att = SelfAttention(self.args.model_args.latent_dim).to(device)


    def forward(self, feature1, feature2):
        f1_cro = self.cross_att(feature1, feature2, feature2) # (5, 256, 256)
        f2_cro = self.cross_att(feature2, feature1, feature1) # (5, 256, 256)

        f1_self = self.self_att(f1_cro.to(device)) # (5, 256, 256)
        f2_self = self.self_att(f2_cro.to(device)) # (5, 256, 256)

        data_cat = torch.cat([f1_self, f2_self], dim=2) # (5, 256, 512)

        data_avg = avg_pooling(data_cat.permute(0, 2, 1), self.args.model_args.latent_dim).permute(0, 2, 1) # (5, 256, 256)

        return data_avg

def avg_pooling(features, target_length):
    batch_size, seq_length, feature_dim = features.size()

    pool = nn.AdaptiveAvgPool1d(target_length)

    features = features.transpose(1, 2).reshape(batch_size * feature_dim, seq_length)

    pooled_features = pool(features)

    pooled_features = pooled_features.reshape(batch_size, feature_dim, target_length).transpose(1, 2)

    return pooled_features


def ConstractGraph(self, audio_data, video_data, text_data):
    video_node = avg_pooling(video_data, self.args.bifg.avg_pool_len)  # (5, 256, 256)
    audio_node = avg_pooling(audio_data, self.args.bifg.avg_pool_len)  # (5, 256, 256)
    text_node = text_data

    self.cross_fusion = CrossModalFusion(self.args)
    av_node = self.cross_fusion(audio_node, video_node)
    vt_node = self.cross_fusion(video_node, text_node)
    at_node = self.cross_fusion(audio_node, text_node)

    avt_node = self.cross_fusion(av_node, text_node)
    vta_node = self.cross_fusion(vt_node, audio_node)
    atv_node = self.cross_fusion(at_node, video_node)

    combined_features = torch.cat([audio_node, video_node, text_node,
                                          av_node, vt_node, at_node,
                                          avt_node, vta_node, atv_node], dim=1)

    num_audio = audio_node.shape[1]
    start_a = 0
    end_a = start_a + num_audio - 1

    num_video = video_node.shape[1]
    start_v = end_a + 1
    end_v = start_v + num_video - 1

    num_text = text_node.shape[1]
    start_t = end_v + 1
    end_t = start_t + num_text - 1

    num_av = av_node.shape[1]
    start_av = end_t + 1
    end_av = start_av + num_av - 1

    num_vt = vt_node.shape[1]
    start_vt = end_av + 1
    end_vt = start_vt + num_vt - 1

    num_at = at_node.shape[1]
    start_at = end_vt + 1
    end_at = start_at + num_at - 1

    num_avt = avt_node.shape[1]
    start_avt = end_at + 1
    end_avt = start_avt + num_avt - 1

    num_vta = vta_node.shape[1]
    start_vta = end_avt + 1
    end_vta = start_vta + num_vta - 1

    num_atv = atv_node.shape[1]
    start_atv = end_vta + 1
    end_atv = start_atv + num_atv - 1


    edge_index = []

    # AV <--> [A, V]
    for i in range(start_a, end_a):
        for j in range(start_av, end_av):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(start_v, end_v):
        for j in range(start_av, end_av):
            edge_index.append([i, j])
            edge_index.append([j, i])

    # VT <--> [V, T]
    for i in range(start_v, end_v):
        for j in range(start_vt, end_vt):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(start_t, end_t):
        for j in range(start_vt, end_vt):
            edge_index.append([i, j])
            edge_index.append([j, i])

    # AT <--> [A, T]
    for i in range(start_a, end_a):
        for j in range(start_at, end_at):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(start_t, end_t):
        for j in range(start_at, end_at):
            edge_index.append([i, j])
            edge_index.append([j, i])

    # AVT <--> [AV, T]
    for i in range(start_av, end_av):
        for j in range(start_avt, end_avt):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(start_t, end_t):
        for j in range(start_avt, end_avt):
            edge_index.append([i, j])
            edge_index.append([j, i])

    # VTA <--> [VT, A]
    for i in range(start_vt, end_vt):
        for j in range(start_vta, end_vta):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(start_a, end_a):
        for j in range(start_vta, end_vta):
            edge_index.append([i, j])
            edge_index.append([j, i])

    # ATV <--> [AT, V]
    for i in range(start_at, end_at):
        for j in range(start_atv, end_atv):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(start_v, end_v):
        for j in range(start_atv, end_atv):
            edge_index.append([i, j])
            edge_index.append([j, i])


    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    self.gat_layer = GATConv(in_channels=self.args.bifg.avg_pool_len, out_channels=self.args.bifg.avg_pool_len,
                             heads=1).to(device)
    output = []
    for i in range(combined_features.shape[0]):
        node_features = combined_features[i]  # 取出当前批次的特征
        data = Data(x=node_features, edge_index=edge_index)
        out = self.gat_layer(data.x.to(device), data.edge_index.to(device))  # 应用 GAT
        output.append(out)

    return torch.stack(output)