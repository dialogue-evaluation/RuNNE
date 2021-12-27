from argparse import ArgumentParser
import os
import json


USE_TYPE_INDS = [25, 41]


def load_jsonl(in_path, subdir):
    data = []
    fname = os.path.join(in_path, subdir, 'test.jsonl')
    with open(fname, 'r') as f:
        for line in f:
            if line.strip() != '':
                data.append(json.loads(line))
    return data


def load_ner_types(in_path, subdir):
    fname = os.path.join(in_path, subdir, 'ners.txt')
    with open(fname, 'r') as f:
        return f.read().split('\n')


class Evaluator:
    def __init__(self, in_path):
        self.eval_data = load_jsonl(in_path, 'ref')
        self.pred_data = load_jsonl(in_path, 'res')
        self.ner_types = load_ner_types(in_path, 'ref')
        self.num_types = len(self.ner_types)
        self.ner_maps = {ner: (i + 1) for i, ner in enumerate(self.ner_types)}

    def evaluate(self):
        tp, fn, fp = 0, 0, 0
        sub_tp, sub_fn, sub_fp = [0] * self.num_types, [0] * self.num_types, [0] * self.num_types
        for gold_example, pred_example in zip(self.eval_data, self.pred_data):
            # gold_ners = set([(sid, s, e, self.ner_maps[t])
            #                  for sid, ner in enumerate(gold_example['ners']) for s, e, t in ner])
            # pred_ners = set([(sid, s, e, self.ner_maps[t])
            #                  for sid, ner in enumerate(pred_example['ners']) for s, e, t in ner])
            gold_ners = set([(s, e, self.ner_maps[t]) for s, e, t in gold_example['ners']])
            pred_ners = set([(s, e, self.ner_maps[t]) for s, e, t in pred_example['ners']])
            tp += len(gold_ners & pred_ners)
            fn += len(gold_ners - pred_ners)
            fp += len(pred_ners - gold_ners)
            for i in range(self.num_types):
                sub_gm = set((s, e) for s, e, t in gold_ners if t == i+1)
                sub_pm = set((s, e) for s, e, t in pred_ners if t == i+1)
                sub_tp[i] += len(sub_gm & sub_pm)
                sub_fn[i] += len(sub_gm - sub_pm)
                sub_fp[i] += len(sub_pm - sub_gm)
        m_r = 0 if tp == 0 else float(tp) / (tp+fn)
        m_p = 0 if tp == 0 else float(tp) / (tp+fp)
        m_f1 = 0 if m_p == 0 else 2.0*m_r*m_p / (m_r+m_p)
        print("Mention F1: {:.2f}%".format(m_f1 * 100))
        print("Mention recall: {:.2f}%".format(m_r * 100))
        print("Mention precision: {:.2f}%".format(m_p * 100))
        print("****************SUB NER TYPES********************")
        f1_scores_list = []
        for i in range(self.num_types):
            sub_r = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fn[i])
            sub_p = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fp[i])
            sub_f1 = 0 if sub_p == 0 else 2.0 * sub_r * sub_p / (sub_r + sub_p)
            f1_scores_list.append(sub_f1)
            # print("{} F1: {:.2f}%".format(self.ner_types[i], sub_f1 * 100))
            # print("{} recall: {:.2f}%".format(self.ner_types[i],sub_r * 100))
            # print("{} precision: {:.2f}%".format(self.ner_types[i],sub_p * 100))
        summary_dict = {}
        summary_dict["Mention F1"] = m_f1
        summary_dict["Mention recall"] = m_r
        summary_dict["Mention precision"] = m_p
        summary_dict["Macro F1"] = sum([each for i, each in enumerate(f1_scores_list) if i in USE_TYPE_INDS]) \
            / float(self.num_types)
        return summary_dict, summary_dict["Macro F1"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = Evaluator(args.in_path)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    _, f1 = evaluator.evaluate()
    with open(os.path.join(args.out_path, 'scores.txt'), 'w') as f:
        f.write('f1_score: {:.5f}'.format(f1))


if __name__ == '__main__':
    main()
