import json

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence


class MedicalNerDataset(Dataset):

    def __init__(self):
        super(MedicalNerDataset, self).__init__()
        with open("./datasets/medical_ner.json", encoding='utf-8') as f:
            self.datasets = json.load(f)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MedicalNerCollateFn(object):

    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        target_list = []
        for item in batch:
            inputs = self.tokenizer(' '.join(item['input']))
            target = torch.LongTensor(item['target'])
            input_ids = torch.LongTensor(inputs['input_ids'][1:-1])
            attention_mask = torch.LongTensor(inputs['attention_mask'][1:-1])
            if len(input_ids) != len(target):
                print("Abnormal sample:", ''.join(item['input']))
                continue

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            target_list.append(target)

        inputs = {
            "input_ids": pad_sequence(input_ids_list, batch_first=True),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True),
        }
        targets = pad_sequence(target_list, batch_first=True)
        return inputs, targets


def create_dataloader(args):
    dataset = MedicalNerDataset()
    valid_size = int(len(dataset) * args.valid_ratio)
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    collate_fn = MedicalNerCollateFn(args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.workers)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=False,
                              num_workers=args.workers)

    return train_loader, valid_loader
