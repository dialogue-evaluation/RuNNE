import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nltk.data import load
from tqdm.auto import tqdm

import os

def collate_to_max_length(batch):
    """
    Расширение всех полей элементов до максимального в батче.

    Параметры:
    ---------------
        batch: батч, где каждый из примеров содержит список следующих элементов:
        [
           tokens : torch.LongTensor of shape (sequence_length) - токены последовательности после токенизации
           type_ids : torch.LongTensor of shape (sequence_length)
           labels_ids : torch.LongTensor of shape (sequence_length) - id сущностей в последовательности
           offsets : torch.LongTensor of shape (sequence_length, 2) - (start, end) диапазоны токенов в исходной последовательности
           id : torch.LongTensor of shape (1) - id контекста (предложения)
           context : str - предложение
           filename : str - имя файла, откуда взято предложение
           txtdata : str - весь текст
           tid : int - id всего текста по public_data
           c_start : int - символьная позиция начала предложения в тексте
           c_end : int - символьная позиция конца предложения в тексте
        ]
    Возвращает:
    ---------------
        output: список расширенных примеров батча, shape каждого из которых [batch, max_length] (для offsets - [batch, max_length, 2])
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(3):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_output = torch.full([batch_size, max_length, 2], 0, dtype=batch[0][3].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_output[sample_idx][: data.shape[0]][:] = data
    output.append(pad_output)

    output.append(torch.stack([x[-7] for x in batch]))
    output.append([x[-6] for x in batch])
    output.append([x[-5] for x in batch])
    output.append([x[-4] for x in batch])
    output.append([x[-3] for x in batch])
    output.append([x[-2] for x in batch])
    output.append([x[-1] for x in batch])

    return output

def brat2data(sentence_tokenizer, format_path, dataset_path):
    """
    Преобразование данных brat в список словарей с сущностями. 

    Параметры:
    ---------------
    sentence_tokenizer
        Токенизатор по предложениям, разбивающий текст на последовательности. 
    format_path : str
        Путь к файлу из public_data, составляющими отображение из текстов в id
    dataset_path : str
        Путь к директории с данными в формате brat. 

    Возвращает:
    ---------------
        context_groups : List[ContextDict]

        ContextDict: 
            "id" : int - id контекста (предложения) 
            "context" : str - контекст (предложение),
            "filename" : str - имя файла
            "txtdata" : str - весь текст
            "tid" : int - id всего текста по public_data
            "c_start" : int - символьная позиция начала предложения в тексте
            "c_end" : int - символьная позиция конца предложения в тексте
            "entities" : List[EntityDict] - все сущности предложения
        EntityDict:
             "tag" : str - имя сущности
             "start" : int - символьная позиция начала сущности в предложении
             "end" : int - символьная позиция начала сущности в предложении
             "eid" : int - порядковый номер сущности в предложении
             "span" : str - сама сущность (подпоследовательность символов)

    """


    # Считываем отображение текстов в их id по public_data
    with open(format_path, "r", encoding = 'UTF-8') as format_file:
        txtdata_to_id = {}
        for line in format_file:
            if line.strip() != '':
                line_map = json.loads(line)
                text = line_map["sentences"]
                tid = line_map["id"]
                txtdata_to_id[text] = tid


    # Начинаем читать файлы из директории с данными
    context_groups = []
    c_idx = 0

    for ad, dirs, files in os.walk(dataset_path):
        for f in tqdm(files):

            if f[-4:] == '.txt':

                with open(dataset_path + '/' + f, "r", encoding='UTF-8') as txtfile:
                    txtdata = txtfile.read()

                try:
                    tid = txtdata_to_id[txtdata]
                except KeyError:
                    continue # Такие файлы пропускаем

                try:
                    annfile = open(dataset_path + '/' + f[:-4] + ".ann", "r", encoding='UTF-8')

                    file_entities = []

                    # Шаг 1. Считываем все сущности из файла аннотации
                    for line in annfile:
                        line_tokens = line.split()
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                            try:
                                file_entities.append({ 
                                    "tag" : line_tokens[1], 
                                    "start" : int(line_tokens[2]),
                                    "end" : int(line_tokens[3]) - 1, # Все конечные позиции на 1 больше, чем действительный конец сущности в тексте
                                })
                            except ValueError:
                                pass # Все неподходящие сущности

                    annfile.close()

                    # Шаг 2. В каждом файле выделить контексты отдельно друг от друга.

                    sentence_spans = sentence_tokenizer.span_tokenize(txtdata)

                    for span in sentence_spans:

                        span_start, span_end = span
                        context = txtdata[span_start : span_end]

                        sentence_entities = [e for e in file_entities if e["start"] >= span_start and e["end"] <= span_end]

                        # Смещаем все позиции сущностей относительно каждого контекста
                        for entity in sentence_entities:

                            entity["start"] = entity["start"] - span_start
                            entity["end"] = entity["end"] - span_start

                        eid = 0
                        simple_entities = []

                        # Сохраняем все сущности
                        for entity in sentence_entities:

                            tag = entity["tag"]
                            start = entity["start"]
                            end = entity["end"]

                            simple_entities.append({
                                "tag" : tag,
                                "start" : start,
                                "end" : end,
                                "eid" : eid,
                                "span" : context[start : end + 1]
                            })

                            eid += 1

                        context_groups.append({
                            "id" : c_idx,
                            "context" : context,
                            "filename" : f[:-4],
                            "txtdata" : txtdata,
                            "tid" : tid,
                            "c_start" : span_start,
                            "c_end": span_end,
                            "entities" : simple_entities
                        })  

                        c_idx += 1 # Перейти к следующему предложению              

                except FileNotFoundError:

                    # print(f"File '{f[:-4]}.ann' not found.")

                    # Файла аннотации не было найдено, создаем "пустой" датасет
                    
                    sentence_spans = sentence_tokenizer.span_tokenize(txtdata)

                    for span in sentence_spans:

                        span_start, span_end = span
                        context = txtdata[span_start : span_end]

                        context_groups.append({
                            "id" : c_idx,
                            "context" : context,
                            "filename" : f[:-4],
                            "txtdata" : txtdata,
                            "tid" : tid,
                            "c_start" : span_start,
                            "c_end": span_end,
                            "entities" : []
                        })

    return context_groups

class IOBESFlatRuNNEDataset(Dataset):
    """
    Датасет RuNNE с самыми внутренними (плоскими) внутренними сущностями, представляемые в формате IOBES (из данных brat).
    На основе torch.utils.Dataset.

    Возвращает элементы в формате   
    item = 
    [
       tokens : torch.LongTensor of shape (sequence_length) - токены последовательности после токенизации
       type_ids : torch.LongTensor of shape (sequence_length)
       labels_ids : torch.LongTensor of shape (sequence_length) - id сущностей в последовательности
       offsets : torch.LongTensor of shape (sequence_length, 2) - (start, end) диапазоны токенов в исходной последовательности
       id : torch.LongTensor of shape (1) - id контекста (предложения)
       context : str - предложение
       filename : str - имя файла, откуда взято предложение
       txtdata : str - весь текст
       tid : int - id всего текста по public_data
       c_start : int - символьная позиция начала предложения в тексте
       c_end : int - символьная позиция конца предложения в тексте
    ]
    
    Аргументы для инициализации:
    ----------------
    dataset_name : str
        Название датасета (для сохранения в <dataset_name>.jsonl файл)
    dataset_path : str
        Путь к директории с данными в формате brat. 
    ners_path : str
        Путь к файлу ners.txt с именами всех видов сущностей.
    format_path : str
        Путь к файлу из public_data, составляющими отображение из текстов в id
    in_path : str
        Путь к директории по формату Evaluator с ref, res и т.п.
    tokenizer
        Токенайзер по схеме WordPiece.
    max_length : int
        Максимальная длина предложений. 
    ----------------
    """

    def __init__(self, dataset_name, dataset_path, ners_path, format_path, in_path, tokenizer, max_length):

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Шаг 1. Считываем все данные в формате brat в список словарей, удобным для обработки. 

        print(f"Loading {dataset_name}:")
        self.all_data = brat2data(load("tokenizers/punkt/russian.pickle"), format_path, dataset_path)

        # Шаг 2. Сохраняем ground-truth данные в <in_path>/ref/<dataset_name>.jsonl 

        lines = []
        all_tids = sorted(list(set([c["tid"] for c in self.all_data])))
        for tid in all_tids:

            entities = []
            for context_group in [data for data in self.all_data if data["tid"] == tid]:
                entities.extend([(e["start"], e["end"], e["tag"]) for e in context_group["entities"]])

            lines.append(json.dumps({"id" : context_group["tid"], "ners" : entities}, ensure_ascii = False) + '\n')

        with open(in_path + f'/ref/{dataset_name}.jsonl', 'w', encoding="utf-8") as f:
            f.writelines(lines)

        # Шаг 3: Фильтруем все сущности, оставляя только плоские (самые внутренние)

        flat_data = []

        for c in self.all_data:

            flats = []

            for ec in c["entities"]:

                start = ec["start"]
                end = ec["end"]

                if len([e for e in c["entities"] if e["start"] >= start and e["end"] < end or e["start"] > start and e["end"] <= end]) == 0:

                    flats.append(ec)

            flat_data.append({**c, "entities" : flats})

        self.flat_data = flat_data

        # Шаг 4. Токенизация по WordPiece и валидация всех сущностей (проверка на ошибки)

        new_data = []
        for c in self.flat_data:

            context = c["context"]

            encodings = self.tokenizer.encode(context)
            tokens = encodings.ids
            type_ids = encodings.type_ids
            offsets = encodings.offsets

            origin_offset2token_idx_start = {}
            origin_offset2token_idx_end = {}
            for token_idx in range(len(tokens)):

                token_start, token_end = offsets[token_idx]
                # skip [CLS] or [SEP]
                if token_start == token_end == 0:
                    continue

                token_end -= 1

                origin_offset2token_idx_start[token_start] = token_idx
                origin_offset2token_idx_end[token_end] = token_idx
            
            valid_entities = []
            for e in c["entities"]:

                start = e["start"]
                end = e["end"]

                try:
                    new_start = origin_offset2token_idx_start[start]
                    new_end = origin_offset2token_idx_end[end]
                except KeyError:
                    # Пример некорректен из-за опечатки, обнуляем наличие сущности (пропускаем)
                    continue

                valid_entities.append({
                        **e,
                        "tl_start" : new_start,
                        "tl_end" : new_end
                    })
            new_data.append({
                    **c,
                    "entities" : valid_entities,
                    "tokens" : tokens,
                    "type_ids" : type_ids,
                    "offsets" : offsets,
                })

        self.flat_data = new_data

        # Шаг 5: Строим последовательность меток по формату IOBES

        labeled_data = []
        for c in self.flat_data:

            labels = []

            for e in c["entities"]:
                labels.append((e["tag"], e["tl_start"], e["tl_end"]))

            label_seq = ["O"] * len(c["tokens"])

            ### iobes

            for tag, start, end in labels:

                if start == end: # S
                    label_seq[start] = 'S-' + tag
                else: # BI*E
                    label_seq[start] = 'B-' + tag 
                    label_seq[end] = 'E-' + tag 
                    for i in range(start + 1, end):
                        label_seq[i] = 'I-' + tag

            labeled_data.append({
                    **c,
                    "labels" : label_seq
                })

        self.data = labeled_data

        # Шаг 6. Загружаем все типы сущностей и присваиваем им id

        with open(ners_path, "r", encoding='UTF-8') as ners_file:

            tags = [t.strip() for t in ners_file.read().split('\n')]

            tag_to_id = {}
            for idx, tag in enumerate(tags):
                tag_to_id['B-' + tag] = idx * 4 + 1
                tag_to_id['I-' + tag] = idx * 4 + 2
                tag_to_id['E-' + tag] = idx * 4 + 3
                tag_to_id['S-' + tag] = idx * 4 + 4
            tag_to_id['O'] = 0

            self.tag_to_id = tag_to_id

        ###

    def __len__(self):

        return len(self.data)

    def __getitem__(self, item):

        data = self.data[item]

        assert len(data["tokens"]) == len(data["type_ids"])
        assert len(data["type_ids"]) == len(data["labels"])

        labels_ids = [self.tag_to_id[tag] for tag in data["labels"]]

        tokens = data["tokens"]
        type_ids = data["type_ids"]
        offsets = data["offsets"]

        context = data["context"]
        filename = data["filename"]
        txtdata = data["txtdata"]
        tid = data["tid"]
        c_start = data["c_start"]
        c_end = data["c_end"]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        labels_ids = labels_ids[: self.max_length]
        offsets = offsets[: self.max_length]

        sep_token = self.tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(labels_ids),
            torch.LongTensor(offsets),
            torch.LongTensor([data["id"]]),
            context,
            filename,
            txtdata,
            tid,
            c_start,
            c_end
        ]


def test_dataset():

    dataset = IOBESFlatRuNNEDataset(dataset_name = "dev",
                                    dataset_path = "../data/dev", 
                                    ners_path = "./eval/ref/ners.txt",
                                    format_path = "../public_data/dev.jsonl", 
                                    in_path = "eval",
                                    tokenizer = BertWordPieceTokenizer("vocab.txt", lowercase = False),
                                    max_length = 128)

    # print(dataset[50])
    print(len(dataset))
    print(dataset[10])
    # filenames = set([dataset[i][-5] for i in range(len(dataset))])
    # print(len(filenames))

    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     collate_fn=collate_to_max_length
    # )

    # for batch in dataloader:
    #     print(batch)
    #     break

if __name__ == '__main__':
    test_dataset()