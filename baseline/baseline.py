import pytorch_lightning as pl
import torch
import json
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup, logging
from tokenizers import BertWordPieceTokenizer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from iobes_flat_dataset import IOBESFlatRuNNEDataset, collate_to_max_length
from score import Evaluator

class BaselineRuBERT(pl.LightningModule):
    """
    BaselineRuBERT - базовый класс модели. 

    Параметры:
    ---------------
    in_path : str
        Путь к директории по формату Evaluator с ref, res и т.п.
    out_path : str
        Путь к директории вывода, куда будут сохраняться последние результаты по метрикам (файл last_scores.txt).
    tag_to_id : Dict[str, int]
        Отображение имён сущностей в формате IOBES по их именам в порядковые номера. Требуется для вывода по Evaluator.
    total_steps : int
        Число шагов для расписания OneCycleLR из torch.optim.scheduler.
    lr : float (default = 1e-4)
        Скорость обучения. 
    weight_decay : float (default = 0.02)
        Сокращение весов, для RuBERT.

    """

    def __init__(
            self,
            in_path,
            out_path,
            tag_to_id,
            total_steps,
            lr = 1e-4,
            weight_decay = 0.02):

        super().__init__()

        # Используется RuBERT от DeepPavlov
        self.model = BertForTokenClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels = 29 * 4 + 1, return_dict = False)

        self.lr = lr
        self.total_steps = total_steps
        self.weight_decay = weight_decay

        self.tag_to_id = tag_to_id

        # Создаем обратное отображение id в имена сущностей
        tags = [None] * (max(self.tag_to_id.values()) + 1)
        for tag, idx in self.tag_to_id.items():
            tags[idx] = tag

        self.id_to_tag = tags

        self.in_path = in_path
        self.out_path = out_path

    def configure_optimizers(self):
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.999), 
                              lr=self.lr,
                              eps=1e-6)

        t_total = self.total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, pct_start=0.3,
            total_steps=t_total, anneal_strategy='linear'
        )

        return {"optimizer" : optimizer, "lr_scheduler" : scheduler}

    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):

        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels = labels)

    def training_step(self, batch, batch_idx):

        # batch:
        # [ 
        #     torch.LongTensor(tokens),
        #     torch.LongTensor(type_ids),
        #     torch.LongTensor(labels_ids),
        #     torch.LongTensor(offsets),
        #     torch.LongTensor([data["id"]]),
        #     context,
        #     filename,
        #     txtdata,
        #     tid,
        #     c_start,
        #     c_end
        # ]

        tokens, token_type_ids, labels = batch[0], batch[1], batch[2]

        attention_mask = (tokens != 0).long()
        loss = self(tokens, attention_mask, token_type_ids, labels = labels)[0]

        return {"loss" : loss}

    def training_epoch_end(self, outputs):

        training_loss = sum([float(loss_dict["loss"]) for loss_dict in outputs])
        print("Loss on train: {:.6f}".format(training_loss))

        self.log('training_loss', training_loss)

    def validation_step(self, batch, batch_idx):

        tokens, token_type_ids, labels, offsets, ids, contexts, filenames, txtdatas, tids, c_starts, c_ends = batch

        # print(filenames)

        attention_mask = (tokens != 0).long()
        loss, logits = self(tokens, attention_mask, token_type_ids, labels = labels)

        return { "loss" : loss, 
                 "logits" : logits, 
                 "labels" : labels, 
                 "ids" : ids, 
                 "offsets" : offsets,
                 "contexts" : contexts,
                 "filenames" : filenames,
                 "txtdatas" : txtdatas, 
                 "tids" : tids,
                 "c_starts" : c_starts,
                 "c_ends" : c_ends }

    def validation_epoch_end(self, outputs):

        # logits = (batch_size, seq_len, labels_num)

        # Сохраняем всё
        all_preds = []
        all_labels = []
        all_ids = []
        all_offsets = []
        all_contexts = []
        all_filenames = []
        all_txtdatas = []
        all_tids = []
        all_c_starts = []
        all_c_ends = []

        sum_loss = 0.
        for output in outputs:

            loss = output["loss"]
            logits = output["logits"]
            labels = output["labels"]
            ids = output["ids"]
            offsets = output["offsets"]
            contexts = output["contexts"]
            filenames = output["filenames"]
            txtdatas = output["txtdatas"]
            tids = output["tids"]
            c_starts = output["c_starts"]
            c_ends = output["c_ends"]

            preds = torch.argmax(logits, dim = 2)
            
            # all_loss.append(loss)
            # all_logits.append(logits)

            all_preds.extend(list(torch.split(preds, 1)))
            all_labels.extend(list(torch.split(labels, 1)))
            all_ids.extend(list(torch.split(ids, 1)))
            all_offsets.extend(list(torch.split(offsets, 1)))
            all_contexts.extend(contexts)
            all_filenames.extend(filenames)
            all_txtdatas.extend(txtdatas)
            all_tids.extend(tids)
            all_c_starts.extend(c_starts)
            all_c_ends.extend(c_ends)

            sum_loss += float(loss)

        # all_loss?
        # logits = torch.cat(all_logits, dim = 0)
        
        print("\nLoss on dev: {:.6f}".format(sum_loss))

        self.log('validation_loss', sum_loss)

        # summary = self.compute_score(all_ids, all_preds, all_labels)

        # print(len(set(all_filenames)))

        sorted_zip = sorted(list(zip(all_ids, all_preds, all_labels, all_offsets, all_contexts, all_filenames, \
            all_txtdatas, all_tids, all_c_starts, all_c_ends)), key = lambda x: x[0])

        summary = self.compute_iobes_score(sorted_zip, mode = "dev")

        self.log("mention_f1", summary["Mention F1"])
        self.log("mention_precision", summary["Mention precision"])
        self.log("mention_recall", summary["Mention recall"])
        self.log("macro_f1", summary["Macro F1"])
        self.log("macro_fewshot_f1", summary["Macro F1 few-shot"])

        return {}

    def compute_iobes_score(self, sorted_zip, mode = "test"):

        # (dataset_size, seq_len)

        all_entities = []

        ### Предсказываем сущности в данных и сверяем их с ground-truth на основе Evaluator.

        # Шаг 1. Преобразование id в имена сущностей, вовзращаем позиции по токенам к символьным (в контексте)
        for cid, preds, labels, offsets, context, filename, txtdata, tid, c_start, c_end in sorted_zip:

            preds = list(torch.squeeze(preds))

            tag_spans = [(t_idx, self.id_to_tag[int(pred)]) for t_idx, pred in enumerate(preds)]
            pred_entities = []

            # single:
            singles = [s for s in tag_spans if s[1][:2] == 'S-']
            pred_entities.extend([(t_idx, t_idx, tag[2:]) for t_idx, tag in singles])

            # B-I-E:
            for b_idx, tag_name in [(i, t[2:]) for i, t in tag_spans if t[:2] == 'B-']:

                idx = b_idx + 1
                while idx < len(tag_spans) and tag_spans[idx][1] == 'I-' + tag_name:
                    idx += 1
                if idx < len(tag_spans) and tag_spans[idx][1] == 'E-' + tag_name:
                    pred_entities.append((b_idx, idx, tag_name))

            offsets = list(torch.squeeze(offsets))

            pred_cl_entities = []

            for tl_start, tl_end, tag in pred_entities:

                cl_start = offsets[tl_start][0]
                cl_end = offsets[tl_end][1]
                pred_cl_entities.append((cl_start, cl_end - 1, tag))

            all_entities.append((cid, context, filename, txtdata, tid, c_start, c_end, pred_cl_entities))

        # Шаг 2. Преобразуем символьные позиции в контексте в символьные позиции в целом тексте и сохраняем

        lines = []

        filenames = set([e[2] for e in all_entities])
        # print(len(filenames))

        for filename in filenames:

            file_entities = [e for e in all_entities if e[2] == filename]
            file_entities = sorted(file_entities, key = lambda x : x[4])
            # file_entities[0][1] = file_entities[0][1][:-1] + '\n'

            text_entities = []

            for context_entities in file_entities:

                cid, context, filename, txtdata, tid, c_start, c_end, pred_cl_entities = context_entities

                for cl_start, cl_end, tag in pred_cl_entities:

                    text_entities.append((int(cl_start + c_start), int(cl_end + c_start), tag))
                    
                    try:
                        assert context[cl_start : cl_end + 1] == txtdata[cl_start + c_start : cl_end + 1 + c_start]
                    except AssertionError:
                        print("From context:")
                        print(context[cl_start : cl_end + 1])
                        print("From text:")
                        print(txtdata[cl_start + c_start : cl_end + 1 + c_start])
                        raise AssertionError

            _, _, filename, txtdata, tid, first_sentence_start, first_sentence_end, _ = file_entities[0]

            lines.append(json.dumps({"id" : tid, "ners" : text_entities}, ensure_ascii = False) + '\n')

        # Сохраняем в <in_path>/res/<dataset_name>.jsonl предсказанные сущности
        with open(self.in_path + f"/res/{mode}.jsonl", "w", encoding = "UTF-8") as f:
            f.writelines(lines)

        ### ПОСЛЕ ЭТОГО ЭТАПА МОЖНО ЗАГРУЖАТЬ ПОЛУЧЕННЫЙ ФАЙЛ С ПРЕДСКАЗАННЫМИ СУЩНОСТЯМИ

        # Запускаем скрипт и получаем результаты
        evaluator = Evaluator(self.in_path, mode = "dev")

        evaluator.validate()
        summary, f1, f1_fs = evaluator.evaluate()
        with open(self.out_path + f'/last_scores_{mode}.txt', 'w') as f:
            f.write('f1_score: {:.5f}\n'.format(f1))
            f.write('f1_score_few_shot: {:.5f}'.format(f1_fs))

        self.summary_f1 = summary["Macro F1"]

        return summary

VOCAB_PATH = "./vocab.txt"
NERS_PATH = "./eval/ref/ners.txt"
IN_PATH = "./eval"
OUT_PATH = "./eval"

TRAIN_PATH = "../data/train"
DEV_PATH = "../data/dev"
# TEST_PATH = "./data/test"

TRAIN_IDS_PATH = "../public_data/train.jsonl"
DEV_IDS_PATH = "../public_data/dev.jsonl"
# TEST_IDS_PATH = "./data/test.jsonl"

CKPT_PATH = "./checkpoints"
CKPT_FILE = "./checkpoints/epoch=206-step=37466.ckpt"

MAX_LEN = 128
BATCH_SIZE = 1
NUM_WORKERS = 8
MAX_EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 0.02

def main():

    logging.set_verbosity_error()

    bertwptokenizer = BertWordPieceTokenizer(VOCAB_PATH, lowercase=False)

    train_dataset = IOBESFlatRuNNEDataset( dataset_name = "train",
                                           dataset_path = TRAIN_PATH, 
                                           ners_path = NERS_PATH, 
                                           format_path = TRAIN_IDS_PATH,
                                           in_path = IN_PATH,
                                           tokenizer = bertwptokenizer, 
                                           max_length = MAX_LEN )

    dev_dataset   = IOBESFlatRuNNEDataset( dataset_name = "dev",
                                           dataset_path = DEV_PATH,   
                                           ners_path = NERS_PATH, 
                                           format_path = DEV_IDS_PATH,
                                           in_path = IN_PATH,
                                           tokenizer = bertwptokenizer, 
                                           max_length = MAX_LEN )

    # filenames = set([dev_dataset[i][-5] for i in range(len(dev_dataset))])
    # print(len(filenames))

    # test_dataset  = IOBESFlatRuNNEDataset( dataset_name = "test", 
    #                                        dataset_path = TEST_PATH,  
    #                                        ners_path = NERS_PATH, 
    #                                        format_path = TEST_IDS_PATH,
    #                                        in_path = IN_PATH,
    #                                        tokenizer = bertwptokenizer, 
    #                                        max_length = MAX_LEN )

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS,
        collate_fn = collate_to_max_length
    )

    dev_dataloader = DataLoader(
        dataset = dev_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        collate_fn = collate_to_max_length
    )

    # test_dataloader = DataLoader(
    #     dataset = test_dataset,
    #     batch_size = BATCH_SIZE,
    #     shuffle = False,
    #     num_workers = NUM_WORKERS,
    #     collate_fn = collate_to_max_length
    # )

    model = BaselineRuBERT (
        in_path = IN_PATH,
        out_path = OUT_PATH,
        tag_to_id = train_dataset.tag_to_id,
        total_steps = (len(train_dataset) // BATCH_SIZE) * MAX_EPOCHS,
        lr = LR,
        weight_decay = WEIGHT_DECAY
    )

    checkpoint_callback = ModelCheckpoint(
        # Директория, куда будут сохраняться чекпойнты и логи (по умолчанию корневая папка проекта)
        dirpath = CKPT_PATH,
        save_top_k = 1,
        verbose = True,
        monitor = 'macro_f1',
        mode = "max", # Сохраняем самые максимальные по метрике модели
    )

    # Настройка сохранения моделей через callbacks
    trainer = Trainer(
        # gpus = -1,
        callbacks = [checkpoint_callback],
        num_sanity_val_steps = -1,
        max_epochs = 1
    )

    trainer.fit(model, train_dataloader, dev_dataloader) # Запуск процесса обучения и валидации, с мониторингом
    


def validate_checkpoint():

    logging.set_verbosity_error()

    bertwptokenizer = BertWordPieceTokenizer(VOCAB_PATH, lowercase=False)

    dev_dataset   = IOBESFlatRuNNEDataset( dataset_name = "dev",
                                           dataset_path = DEV_PATH,   
                                           ners_path = NERS_PATH, 
                                           format_path = DEV_IDS_PATH,
                                           in_path = IN_PATH,
                                           tokenizer = bertwptokenizer, 
                                           max_length = MAX_LEN )

    dev_dataloader = DataLoader(
        dataset = dev_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        collate_fn = collate_to_max_length
    )

    ckpt_model = BaselineRuBERT.load_from_checkpoint(
        CKPT_FILE, 
        in_path = IN_PATH,
        out_path = OUT_PATH,
        tag_to_id = dev_dataset.tag_to_id,
        total_steps = 0, 
        lr = LR,
        weight_decay = WEIGHT_DECAY
    )

    trainer = Trainer(gpus = -1)
    trainer.validate(ckpt_model, dataloaders = dev_dataloader)

# Для обучения запусаем main(), для тестирования чекпойнта - validate_checkpoint(). 
# Все параметры меняем по усмотрению для удобства и контроля обучения и/или тестирования. 

if __name__ == '__main__':
    main()
    # validate_checkpoint()