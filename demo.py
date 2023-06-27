from transformers import AutoModelForTokenClassification, BertTokenizerFast

from models.base import MedicalNerModel

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('iioSnail/bert-base-chinese-medical-ner')
    model = AutoModelForTokenClassification.from_pretrained("iioSnail/bert-base-chinese-medical-ner")

    sentences = ["瘦脸针、水光针和玻尿酸详解！", "半月板钙化的病因有哪些？"]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=False)
    outputs = model(**inputs)
    outputs = outputs.logits.argmax(-1) * inputs['attention_mask']

    print(outputs)
    print("-" * 20)
    preds = MedicalNerModel.format_outputs(sentences, outputs)
    print(preds)