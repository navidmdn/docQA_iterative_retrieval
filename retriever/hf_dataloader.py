import datasets
from typing import List, Dict, Tuple

import transformers
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import torch
import random
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dev_path: str = field(default=None, metadata={"help": "Path to the dev dataset."})
    test_path: str = field(default=None, metadata={"help": "Path to the test datasets."})
    batch_size: int = field(default=32, metadata={"help": "Batch size."})
    num_workers: int = field(default=1, metadata={"help": "Number of workers for dataloader."})
    max_c_len: int = field(default=300, metadata={"help": "Max context length."})
    max_q_len: int = field(default=70, metadata={"help": "Max question length."})
    max_q_sp_len: int = field(default=350, metadata={"help": "Max question + supporting context length."})


class DataModule:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 train_path: str,
                 test_path: str,
                 dev_path: str,
                 batch_size: int,
                 num_workers: int = 8,
                 max_c_len: int = 300,
                 max_q_len: int = 70,
                 max_q_sp_len: int = 350,
                 cache_dir: str = None
                 ):
        self.train_path = train_path
        self.test_path = test_path
        self.dev_path = dev_path
        self.batch_size = batch_size
        self.max_c_len = max_c_len
        self.cache_dir = cache_dir
        self.max_q_len = max_q_len
        self.max_q_sp_len = max_q_sp_len
        self.num_workers = num_workers
        self.tokenizer = tokenizer

    @staticmethod
    def collate_tokens(samples, pad_id):
        if len(samples) == 0:
            return {}

        if len(samples[0].size()) > 1:
            samples = [v.view(-1) for v in samples]

        max_len = max([len(s) for s in samples])
        batch = []
        for s in samples:
            batch.append(torch.cat([s, torch.ones(max_len - len(s), dtype=torch.long).to(s) * pad_id], dim=0))

        return torch.stack(batch, dim=0)

    def mhop_collate(self, samples):
        dict_of_lists = {}
        for k in samples[0].keys():
            dict_of_lists[k] = [torch.tensor(s[k]) for s in samples]

        batch = {
            'q_input_ids': self.collate_tokens(dict_of_lists["q_input_ids"], pad_id=self.tokenizer.pad_token_id),
            'q_mask': self.collate_tokens(dict_of_lists["q_mask"], pad_id=self.tokenizer.pad_token_id),
            'q_sp_input_ids': self.collate_tokens(dict_of_lists["q_sp_input_ids"], pad_id=self.tokenizer.pad_token_id),
            'q_sp_mask': self.collate_tokens(dict_of_lists["q_sp_mask"], pad_id=self.tokenizer.pad_token_id),
            'c1_input_ids': self.collate_tokens(dict_of_lists["c1_input_ids"], pad_id=self.tokenizer.pad_token_id),
            'c1_mask': self.collate_tokens(dict_of_lists["c1_mask"], pad_id=self.tokenizer.pad_token_id),
            'c2_input_ids': self.collate_tokens(dict_of_lists["c2_input_ids"], pad_id=self.tokenizer.pad_token_id),
            'c2_mask': self.collate_tokens(dict_of_lists["c2_mask"], pad_id=self.tokenizer.pad_token_id),
            'neg1_input_ids': self.collate_tokens(dict_of_lists["neg1_input_ids"], pad_id=self.tokenizer.pad_token_id),
            'neg1_mask': self.collate_tokens(dict_of_lists["neg1_mask"], pad_id=self.tokenizer.pad_token_id),
            'neg2_input_ids': self.collate_tokens(dict_of_lists["neg2_input_ids"], pad_id=self.tokenizer.pad_token_id),
            'neg2_mask': self.collate_tokens(dict_of_lists["neg2_mask"], pad_id=self.tokenizer.pad_token_id),
            # 'q_type_ids': self.collate_tokens(dict_of_lists["q_type_ids"], pad_id=self.tokenizer.pad_token_id),
            # 'q_sp_type_ids': self.collate_tokens(dict_of_lists["q_sp_type_ids"], pad_id=self.tokenizer.pad_token_id),
            # 'c1_type_ids': self.collate_tokens(dict_of_lists["c1_type_ids"], pad_id=self.tokenizer.pad_token_id),
            # 'c2_type_ids': self.collate_tokens(dict_of_lists["c2_type_ids"], pad_id=self.tokenizer.pad_token_id),
            # 'neg1_type_ids': self.collate_tokens(dict_of_lists["neg1_type_ids"], pad_id=self.tokenizer.pad_token_id),
            # 'neg2_type_ids': self.collate_tokens(dict_of_lists["neg2_type_ids"], pad_id=self.tokenizer.pad_token_id),
        }

        return batch

    def encode_pair_sentences(self, sent1s: List[str], sent2s: List[str] = None, max_len: int = 100):
        return self.tokenizer(sent1s, text_pair=sent2s, max_length=max_len,
                              return_tensors="pt", truncation=True, padding='longest')

    def preprocess_data(self, samples: Dict) -> Dict:
        samples = [{k: samples[k][i] for k in samples} for i in range(len(samples["question"]))]
        for sample in samples:
            start_para = bridge_para = None

            if sample['question'].endswith("?"):
                sample['question'] = sample['question'][:-1]
            # for comparison samples, we randomly shuffle the positive paragraphs
            if sample["type"] == "comparison":
                random.shuffle(sample["pos_paras"])
                start_para, bridge_para = sample["pos_paras"]
            else:
                # we assume we only have 2hop samples
                for para in sample["pos_paras"]:
                    if para["title"] != sample["bridge"]:
                        start_para = para
                    else:
                        bridge_para = para
            random.shuffle(sample["neg_paras"])
            sample["start_para"] = start_para
            sample["bridge_para"] = bridge_para
            assert start_para is not None and bridge_para is not None

        start_paras = [s["start_para"] for s in samples]
        bridge_paras = [s["bridge_para"] for s in samples]
        neg_paras_1 = [s["neg_paras"][0] for s in samples]
        neg_paras_2 = [s["neg_paras"][1] for s in samples]
        questions = [s["question"] for s in samples]

        start_para_codes = self.encode_pair_sentences(
            sent1s=[s["title"].strip() for s in start_paras],
            sent2s=[s["text"].strip() for s in start_paras],
            max_len=self.max_c_len
        )

        bridge_para_codes = self.encode_pair_sentences(
            sent1s=[s["title"].strip() for s in bridge_paras],
            sent2s=[s["text"].strip() for s in bridge_paras],
            max_len=self.max_c_len
        )
        neg_codes_1 = self.encode_pair_sentences(
            sent1s=[s["title"].strip() for s in neg_paras_1],
            sent2s=[s["text"].strip() for s in neg_paras_1],
            max_len=self.max_c_len
        )
        neg_codes_2 = self.encode_pair_sentences(
            sent1s=[s["title"].strip() for s in neg_paras_2],
            sent2s=[s["text"].strip() for s in neg_paras_2],
            max_len=self.max_c_len
        )
        q_sp_codes = self.encode_pair_sentences(
            sent1s=questions,
            sent2s=[s["title"]+'\n'+s["text"] for s in start_paras],
            max_len=self.max_q_sp_len
        )

        q_codes = self.encode_pair_sentences(
            sent1s=questions,
            max_len=self.max_q_len
        )

        return {
            'q_input_ids': q_codes["input_ids"],
            'q_mask': q_codes["attention_mask"],
            'q_sp_input_ids': q_sp_codes["input_ids"],
            'q_sp_mask': q_sp_codes["attention_mask"],
            'c1_input_ids': start_para_codes["input_ids"],
            'c1_mask': start_para_codes["attention_mask"],
            'c2_input_ids': bridge_para_codes["input_ids"],
            'c2_mask': bridge_para_codes["attention_mask"],
            'neg1_input_ids': neg_codes_1["input_ids"],
            'neg1_mask': neg_codes_1["attention_mask"],
            'neg2_input_ids': neg_codes_2["input_ids"],
            'neg2_mask': neg_codes_2["attention_mask"],
            # 'q_type_ids': q_codes["token_type_ids"],
            # 'c1_type_ids': start_para_codes["token_type_ids"],
            # 'c2_type_ids': bridge_para_codes["token_type_ids"],
            # "q_sp_type_ids": q_sp_codes["token_type_ids"],
            # 'neg1_type_ids': neg_codes_1["token_type_ids"],
            # 'neg2_type_ids': neg_codes_2["token_type_ids"],
        }

    def load_dataset(self):
        data_files = {'train': self.train_path, 'dev': self.dev_path}
        if self.test_path is not None:
            data_files.update({'test': self.test_path})

        raw_dataset = datasets.load_dataset('json', data_files=data_files, cache_dir=self.cache_dir)

        train_ds = raw_dataset['train']
        print(("train_ds size before filtering:", len(train_ds)))
        raw_dataset['train'] = train_ds.filter(lambda x: len(x['neg_paras']) > 1)
        print(("train_ds size after filtering low negative samples:", len(raw_dataset['train'])))

        preprocessed_dataset = raw_dataset.map(self.preprocess_data, batched=True, num_proc=self.num_workers,
                                               remove_columns=['question', 'pos_paras', 'neg_paras', 'bridge',
                                                               'type', 'answers', '_id'])

        return {
            'train_dataset': preprocessed_dataset['train'],
            'eval_dataset': preprocessed_dataset['dev'],
            'data_collator': self.mhop_collate,
        }
