import copy
import torch
import torch.nn as nn
from lib.trainer_extra.acumen_trainer import AcumenTrainer

class ClassificationTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)

        from lib import nomenclature

        self.losses = {
            loss_args.target_head: nomenclature.LOSSES[loss_args.kind](self.args, loss_args = loss_args.args)
            for loss_args in self.args.losses
        }

    def score_convert_category(self, input_tensor):
        output_tensor = torch.zeros_like(input_tensor)
        output_tensor[(input_tensor >= 0) & (input_tensor <= 4)] = 0
        output_tensor[(input_tensor >= 5) & (input_tensor <= 9)] = 1
        output_tensor[(input_tensor >= 10) & (input_tensor <= 14)] = 2
        output_tensor[(input_tensor >= 15) & (input_tensor <= 19)] = 3
        output_tensor[(input_tensor >= 20) & (input_tensor <= 24)] = 4

        return output_tensor

    def training_step(self, batch, batch_idx):
        for key, value in batch.items():
            if 'modality' in key:
                batch[key] = batch[key].squeeze(1)

        outputs = self.model(batch)

        y_true_category = self.score_convert_category(batch['labels'])

        losses = 0
        for head_name, model_output in outputs.items():
            if head_name not in self.losses:
                continue

            loss = self.losses[head_name](y_true = batch['labels'], y_pred = model_output, type='')
            loss_category = self.losses[head_name](y_true = y_true_category, y_pred = model_output, type='cate')

            losses = losses + loss + loss_category

            self.log(f'train/loss:{head_name}', loss.item())

        final_loss = losses

        self.log('train/loss:final', final_loss.item(), on_step = True)
        return final_loss
