import argparse

import lightning.pytorch as pl
import torch.optim
from torch import nn
from transformers import BertTokenizerFast, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class MedicalNerModel(pl.LightningModule):

    def __init__(self, args: argparse.Namespace):
        super(MedicalNerModel, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.bert = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ner')
        self.head = BertOnlyMLMHead(BertConfig(vocab_size=5))

        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, inputs):
        outputs = self.bert(**inputs).last_hidden_state
        outputs = self.head(outputs)

        return outputs

    def compute_loss(self, outputs, targets):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = self.loss_fnt(outputs, targets)

        self.log("train_loss", loss.item(), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, = batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)

        return {
            'loss': loss,
            'outputs': outputs.argmax(-1) * inputs['attention_mask'],
            'targets': targets,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        targets_size = batch[1].size()
        preds = outputs['outputs']
        targets = outputs['targets']

        correct_num = torch.all(preds == targets, dim=1).sum().item()
        total_num = targets_size[0]

        self.log("train_acc", correct_num / total_num, prog_bar=True)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        outputs = self.forward(inputs)

        preds = outputs.argmax(-1) * inputs['attention_mask']

        correct_num = torch.all(preds == targets, dim=1).sum().item()
        total_num = targets.size(0)

        self.log("val_acc", correct_num / total_num)

        return {
            'outputs': preds,
            'targets': targets,
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)
