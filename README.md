# 中文医疗领域的命名实体识别

## 快速上手

Huggingface地址：https://huggingface.co/iioSnail/bert-base-chinese-medical-ner

使用方式：

```python
from transformers import AutoModelForTokenClassification, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('iioSnail/bert-base-chinese-medical-ner')
model = AutoModelForTokenClassification.from_pretrained("iioSnail/bert-base-chinese-medical-ner")

sentences = ["瘦脸针、水光针和玻尿酸详解！", "半月板钙化的病因有哪些？"]
inputs = tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=False)
outputs = model(**inputs)
outputs = outputs.logits.argmax(-1) * inputs['attention_mask']

print(outputs)
```

输出结果：

```
tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 4, 4],
        [1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 0, 0]])
```

其中 1=B, 2=I, 3=E, 4=O。1, 3表示一个二字医疗实体，1,2,3表示一个3字医疗实体, 1,2,2,3表示一个4字医疗实体，依次类推。

可以使用项目中的MedicalNerModel.format_outputs(sentences, outputs)来将输出进行转换。

效果如下：

```
[
  [
    {'start': 0, 'end': 3, 'word': '瘦脸针'},
    {'start': 4, 'end': 7, 'word': '水光针'},
    {'start': 8, 'end': 11, 'word': '玻尿酸'}、
  ],
  [
    {'start': 0, 'end': 5, 'word': '半月板钙化'}
  ]
]
```

## 数据集

本模型训练使用的原始数据集来源：https://github.com/yzhihao/MCSCSet.git

使用了`tools/process_mcsc_data.py`对原始数据集进行了转换。

转换后的数据集下载地址：https://pan.baidu.com/s/1Hfr2Kgm5CBpnGUQIoiXF9g?pwd=0zmi

下载后将其放在`datasets`目录下

# 模型训练

```shell
sh train.sh
```

# 模型性能

暂时并没有对模型进行详细评估，在验证集的精准率大约为76%。并且我发现，非医疗的命名实体也容易被误认为医疗实体。