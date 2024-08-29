import torch
from einops.layers.torch import Reduce

from .repeat import repeat as multi_sequential_repeat
from .perceiver_blocks import TransformerLayer
from lib.model_extra import MultiHead, ModelOutput
import torch.nn as nn
from einops.layers.torch import Rearrange
from .lib import BiFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        from lib import nomenclature

        assert self.args.n_temporal_windows == 1, f"The Baseline Model only supports one temporal window, but instead it was found {self.args.n_temporal_windows} windows"

        self.modality_to_id = {modality.name:id for id, modality in enumerate(sorted(self.args.modalities, key = lambda x: x.name))}

        self.modality_encoders = torch.nn.ModuleDict({
            modality.name: nomenclature.MODALITY_ENCODERS[modality.name](args, modality)
            for modality in self.args.modalities
        })

        self.modality_embeddings = torch.nn.Embedding(
            len(self.args.modalities), self.args.model_args.latent_dim,
        )

        self.MLP_Communicator = MLP_Communicator(self.args.model_args.latent_dim, channel=self.args.batch_size, hidden_size=1560, depth=1)
        self.MLP_Audio_Communicator = MLP_Communicator(self.args.model_args.latent_dim, channel=self.args.batch_size, hidden_size=1200, depth=1)
        self.MLP_Video_Communicator = MLP_Communicator(self.args.model_args.latent_dim, channel=self.args.batch_size, hidden_size=720, depth=1)

        self.cross_attention = CrossModalityAttention(self.args.model_args.latent_dim)



        self.transformer_block = multi_sequential_repeat(
            self.args.model_args.num_layers,
            lambda lnum: TransformerLayer(
                self.args,
            ),
            self.args.model_args.layer_dropout_rate,
        )


        self.classification_layer = MultiHead(args)

    def forward(self, batch, latent = None):
        all_modality_data = []
        all_modality_mask = []
        audio_data = []
        video_data = []
        text_data = []

        framerate_ratio = batch['video_frame_rate'] / batch['audio_frame_rate']

        for modality in self.args.modalities:
            modality_id = modality.name
            modality_name = modality_id.split('_')[1]

            data = batch[f"modality:{modality_id}:data"]
            mask = batch[f"modality:{modality_id}:mask"]

            data = self.modality_encoders[modality_id](data, mask, framerate_ratio = framerate_ratio)

            data = data + self.modality_embeddings(torch.tensor(self.modality_to_id[modality_id]).to(data.device))

            all_modality_data.append(data)
            all_modality_mask.append(mask)

            if modality_name in ['video', 'facial', 'gaze', 'head']:
                video_data.append(data)
            elif modality_name == 'audio':
                audio_data.append(data)
            elif modality_name == 'text':
                text_data.append(data)

        cat_video_data = torch.cat(video_data, dim=1)  # (5, 720, 256)
        cat_audio_data = torch.cat(audio_data, dim=1)  # (5, 1200, 256)
        cat_text_data = torch.cat(text_data, dim=1)    # (5, 180, 256)

        MLP_video_data = self.MLP_Video_Communicator(cat_video_data.permute(1,0,2)).permute(1,0,2) # (5, 720, 256)
        MLP_audio_data = self.MLP_Audio_Communicator(cat_audio_data.permute(1,0,2)).permute(1,0,2) # (5, 1200, 256)
        MLP_data = torch.cat((MLP_video_data, MLP_audio_data), dim=1)  # (5, 1920, 256)

        cross_text = self.cross_attention(cat_text_data, MLP_data, MLP_data)  # (5, 180, 256)

        graph = BiFG.ConstractGraph(self, MLP_audio_data, MLP_video_data, cross_text) # (5, 2304, 256)

        output = graph

        # output: (5, 256)
        output = Reduce('b n d -> b d', 'mean')(output)

        output = ModelOutput(representation = output)

        model_output = self.classification_layer(output)
        model_output['latent'] = None

        return model_output


class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x

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