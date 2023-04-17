import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorWithPadding
from typing import Dict
import random


class RetrieverDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, train_path, dev_path, test_path, max_q_len, max_q_sp_len, max_c_len,):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.train = None
        self.dev = None
        self.test = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.max_c_len = max_c_len
        self.max_q_len = max_q_len
        self.max_q_sp_len = max_q_sp_len
        self.datacollator = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors='pt')

    def encode_paragraph(self, paragraph: Dict, max_len: int):
        return self.tokenizer(paragraph["title"].strip(), text_pair=paragraph["text"].strip(), max_length=max_len,
                              return_tensors="pt")

    def preprocess_data(self, sample: Dict) -> Dict:
        question = sample['question']
        start_para = bridge_para = None

        if question.endswith("?"):
            question = question[:-1]
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
        if self.train:
            random.shuffle(sample["neg_paras"])

        assert start_para is not None and bridge_para is not None

        start_para_codes = self.encode_paragraph(start_para, self.max_c_len)
        bridge_para_codes = self.encode_paragraph(bridge_para, self.max_c_len)
        neg_codes_1 = self.encode_paragraph(sample["neg_paras"][0], self.max_c_len)
        neg_codes_2 = self.encode_paragraph(sample["neg_paras"][1], self.max_c_len)

        q_sp_codes = self.tokenizer(question, text_pair=start_para["text"].strip(),
                                    max_length=self.max_q_sp_len, return_tensors="pt")
        q_codes = self.tokenizer(question, max_length=self.max_q_len, return_tensors="pt")

        return {
            "q_codes": q_codes,
            "q_sp_codes": q_sp_codes,
            "start_para_codes": start_para_codes,
            "bridge_para_codes": bridge_para_codes,
            "neg_codes_1": neg_codes_1,
            "neg_codes_2": neg_codes_2,
        }

    def prepare_data(self):

        train_raw = load_dataset('json', self.train_path, split='train')
        dev_raw = load_dataset('json', self.dev_path, split='dev')
        test_raw = load_dataset('json', self.test_path, split='test')

        raw_dataset = DatasetDict({'train': train_raw, 'dev': dev_raw, 'test': test_raw})

        # should we parallelize this here? since it is a pytorch lightning module
        preprocessed_dataset = raw_dataset.map(self.preprocess_data, batched=True, num_proc=8)

        # save processed dataset
        preprocessed_dataset.save_to_disk(self.args.preprocessed_data_dir)

    def setup(self, stage: str):
        dataset = load_dataset('json', self.args.preprocessed_data_dir)
        if stage == 'fit':
            train_dataset = dataset['train']
            dev_dataset = dataset['dev']

            #todo: check the datacollator we are using
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.datacollator
            )
            self.dev_loader = DataLoader(
                dev_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.datacollator
            )

        if stage == 'test':
            self.test_loader = DataLoader(
                dataset['test'], batch_size=self.args.batch_size, shuffle=False, collate_fn=self.datacollator
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.dev_loader

    def test_dataloader(self):
        return self.test_loader

