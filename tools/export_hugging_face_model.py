import os

import torch

from models.base import MedicalNerModel

os.chdir(os.path.pardir)

def main():
    ckpt_path = "./outputs/best.ckpt"
    output_path = "./outputs/huggingface"

    model = MedicalNerModel(None)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    model.tokenizer.save_pretrained(output_path)
    model.model.save_pretrained(output_path)

    print("Save success!")

if __name__ == '__main__':
    main()
