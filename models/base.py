import argparse

import lightning.pytorch as pl
import torch.optim
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup, \
    AutoModelForTokenClassification


class MedicalNerModel(pl.LightningModule):

    def __init__(self, args: argparse.Namespace):
        super(MedicalNerModel, self).__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=5)

        self.val_correct_num = 0
        self.val_total_num = 0

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets, = batch
        outputs = self.model(**inputs, labels=targets)
        loss = outputs.loss
        outputs = outputs.logits

        self.log("train_loss", loss.item(), prog_bar=True)

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
        outputs = self.model(**inputs).logits

        preds = outputs.argmax(-1) * inputs['attention_mask']

        correct_num = torch.all(preds == targets, dim=1).sum().item()
        total_num = targets.size(0)

        self.log("val_acc", correct_num / total_num)

        self.val_correct_num += correct_num
        self.val_total_num += total_num

        return {
            'outputs': preds,
            'targets': targets,
        }

    def on_validation_epoch_end(self) -> None:
        print("Epoch",self.current_epoch, ". val_acc:", self.val_correct_num / self.val_total_num)
        self.val_correct_num = 0
        self.val_total_num = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)

        t_total = len(self.args.train_loader) * self.args.epochs

        warmup_steps = int(0.1 * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @staticmethod
    def format_outputs(sentences, outputs):
        preds = []
        for i, pred_indices in enumerate(outputs):
            words = []
            start_idx = -1
            end_idx = -1
            flag = False
            for idx, pred_idx in enumerate(pred_indices):
                if pred_idx == 1:
                    start_idx = idx
                    flag = True
                    continue

                if flag and pred_idx != 2 and pred_idx != 3:
                    # 出现了不应该出现的index
                    print("Abnormal prediction results for sentence", sentences[i])
                    start_idx = -1
                    end_idx = -1
                    continue

                if pred_idx == 3:
                    end_idx = idx

                    words.append({
                        "start": start_idx,
                        "end": end_idx + 1,
                        "word": sentences[i][start_idx:end_idx+1]
                    })
                    start_idx = -1
                    end_idx = -1
                    flag = False
                    continue

            preds.append(words)

        return preds
