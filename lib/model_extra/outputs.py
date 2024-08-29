import torch
import torch.nn as nn

from coral_pytorch.dataset import proba_to_label
from coral_pytorch.layers import CoralLayer


from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

class ClassificationOutput(NamedTuple):
    logits: torch.Tensor
    probas: torch.Tensor
    labels: torch.Tensor
    category: torch.Tensor

class RegressionOutput(NamedTuple):
    logits: torch.Tensor
    probas: torch.Tensor
    labels: torch.Tensor

class ModelOutput(NamedTuple):
    representation: torch.Tensor


class MultiHead(torch.nn.Module):
    def __init__(self, args):
        super(MultiHead, self).__init__()

        from lib import nomenclature

        self.args = args

        if not isinstance(self.args.heads, list):
            self.args.heads = [self.args.heads]

        self.heads = torch.nn.ModuleDict({
            head_args.name: nomenclature.HEADS[head_args.kind](args = self.args, head_args = head_args.args)
            for head_args in self.args.heads
        })

    def forward(self, model_output: ModelOutput):
        aggregated_results = {}

        for name, module in self.heads.items():
            module_outputs = module(model_output)
            aggregated_results[name] = module_outputs

        return aggregated_results


class CoralHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(CoralHead, self).__init__()
        self.args = args
        self.head_args = head_args

        self.outputs = CoralLayer(
            size_in = self.args.model_args.latent_dim,
            num_classes = self.head_args.num_classes
        )

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:
        logits = self.outputs(model_output.representation)
        probas = torch.sigmoid(logits)
        labels = proba_to_label(probas).float()

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels
        )

        return output_results

class RegressionHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(RegressionHead, self).__init__()
        self.args = args
        self.head_args = head_args
        self.outputs = nn.Linear(self.args.model_args.latent_dim, self.head_args.num_classes, bias = False)

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:
        logits = self.outputs(model_output.representation)
        probas = torch.nn.functional.softmax(logits, dim=-1)
        labels = logits.argmax(dim=-1)

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels
        )

        return output_results

# class ClassificationHead(torch.nn.Module):
#     def __init__(self, args, head_args = None):
#         super(ClassificationHead, self).__init__()
#         self.args = args
#         self.head_args = head_args
#
#         if args.target == "classification":
#             self.out_dim = self.head_args.num_classes
#         elif args.target == "regression":
#             self.out_dim = self.head_args.score_range
#         else:
#             raise ValueError("args.target must is classification or regression !")
#
#         self.outputs = nn.Linear(self.args.model_args.latent_dim, self.out_dim, bias = False)
#
#     def forward(self, model_output: ModelOutput) -> ClassificationOutput:
#         logits = self.outputs(model_output.representation)
#         probas = torch.nn.functional.softmax(logits, dim = -1)
#         labels = logits.argmax(dim = -1)
#
#         output_results = ClassificationOutput(
#             logits = logits,
#             probas = probas,
#             labels = labels,
#         )
#
#         return output_results

# class MultiLabelHead(torch.nn.Module):
#     def __init__(self, args, head_args = None):
#         super(MultiLabelHead, self).__init__()
#         self.args = args
#         self.head_args = head_args
#         self.outputs = nn.Linear(self.args.model_args.latent_dim, self.head_args.num_classes, bias = False)
#
#
#     def forward(self, model_output: ModelOutput) -> ClassificationOutput:
#         logits = self.outputs(model_output.representation)
#         probas = torch.sigmoid(logits)
#         labels = torch.round(probas)  # 四舍五入
#
#         output_results = ClassificationOutput(
#             logits = logits,
#             probas = probas,
#             labels = labels
#         )
#
#         return output_results

class ClassificationHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(ClassificationHead, self).__init__()
        self.args = args
        self.head_args = head_args

        if args.target == "classification":
            self.out_dim = self.head_args.num_classes
        elif args.target == "regression":
            self.out_dim = self.head_args.score_range
        else:
            raise ValueError("args.target must is classification or regression !")

        self.score_mlp1 = nn.Linear(self.args.model_args.latent_dim, 128, bias = False)
        self.score_mlp2 = nn.Linear(128, 64, bias = False)
        self.score = nn.Linear(64, self.out_dim, bias = False)

        self.category_mlp1 = nn.Linear(self.args.model_args.latent_dim, 128, bias=False)
        self.category_mlp2 = nn.Linear(128, 64, bias=False)
        self.category = nn.Linear(64, 5, bias = False)

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:

        # category predictor
        cate = self.category_mlp1(model_output.representation)
        cate = self.category_mlp2(cate)
        cate_logits = self.category(cate)
        cate = torch.softmax(cate_logits, dim=-1)

        # score predictor
        logits = self.score_mlp1(model_output.representation)
        logits = self.score_mlp2(logits)
        logits = self.score(logits)
        probas = torch.softmax(logits, dim=-1)

        # 扩充cate张量并与score概率加和
        cate_expand = torch.repeat_interleave(cate, 5, dim=-1)
        sum_cate_score = cate_expand + probas
        labels = sum_cate_score.argmax(dim=-1)

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels,
            category = cate_logits
        )

        return output_results

class MultiLabelHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(MultiLabelHead, self).__init__()
        self.args = args
        self.head_args = head_args
        self.score_mlp1 = nn.Linear(self.args.model_args.latent_dim, 128, bias = False)
        self.score_mlp2 = nn.Linear(128, 64, bias = False)
        self.score = nn.Linear(64, self.head_args.num_classes, bias = False)

        self.category_mlp1 = nn.Linear(self.args.model_args.latent_dim, 128, bias=False)
        self.category_mlp2 = nn.Linear(128, 64, bias=False)
        self.category = nn.Linear(64, 5, bias = False)

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:
        # score predictor
        logits = self.score_mlp1(model_output.representation)
        logits = self.score_mlp2(logits)
        logits = self.score(logits)
        probas = torch.sigmoid(logits)
        labels = torch.round(probas)  # 四舍五入

        # category predictor
        cate = self.category_mlp1(model_output.representation)
        cate = self.category_mlp2(cate)
        cate = self.category(cate)
        cate = torch.sigmoid(cate)

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels,
            category = cate
        )

        return output_results
