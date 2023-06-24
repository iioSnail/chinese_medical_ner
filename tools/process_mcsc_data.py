import json


def check_is_valid(string):
    try:
        string.encode('gbk')
        return True
    except UnicodeEncodeError:
        return False


def load_datasets():
    with open("annotated_data.txt", encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    for line in lines:
        items = line.split("\t")
        if len(items) != 3:
            print("Abnormal sample:", line)
            continue

        sent = items[1].replace(" ", "").strip()
        # Clean data: remove the unreadable character like '�'.
        if not check_is_valid(sent):
            print("Abnormal sentence:", sent)
            continue

        sentences.append(sent)

    return sentences


def label_datasets(sentences):
    samples = []
    for sent in sentences:
        sent = sent.replace(" ", "").strip()
        tokens = []
        label = []
        target = []
        flag = False
        for i, token in enumerate(sent):
            if token in ["{", "}"]:
                continue

            tokens.append(token)

            if i > 0 and sent[i - 1] == "{":
                label.append("B")
                target.append(1)
                flag = True
            elif i < len(sent) - 1 and sent[i + 1] == "}":
                label.append("E")
                target.append(3)
                flag = False
            elif flag:
                label.append("I")
                target.append(2)
            else:
                label.append("O")
                target.append(4)

        samples.append({
            "input": tokens,
            "label": label,
            "target": target,
        })

    return samples


def main():
    sentences = load_datasets()
    samples = label_datasets(sentences)
    with open("medical_ner.json", mode='w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False)

    print("Success to process the MCSCSet data!")


if __name__ == '__main__':
    main()
