import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from typing import Dict
import random
import torch


class RetrieverDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_cp: str,
                 train_path: str,
                 dev_path: str,
                 max_c_len: int,
                 max_q_len: int,
                 max_q_sp_len: int,
                 batch_size: int,
                 preprocessed_data_dir: str,
                 num_workers: int = 8,
                 do_test: bool = False,
                 test_path: str = "",
                 device: str = "cpu",
                 train: bool = True,
                 ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cp)
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.device = device
        self.train = train
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.max_c_len = max_c_len
        self.max_q_len = max_q_len
        self.do_test = do_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.preprocessed_data_dir = preprocessed_data_dir
        self.max_q_sp_len = max_q_sp_len

    def encode_paragraph(self, paragraph: Dict, max_len: int):
        return self.tokenizer(paragraph["title"].strip(), text_pair=paragraph["text"].strip(), max_length=max_len,
                              return_tensors="pt", truncation=True)

    @staticmethod
    def collate_tokens(samples, pad_id=0):
        if len(samples) == 0:
            return {}

        if len(samples[0].size()) > 1:
            samples = [v.view(-1) for v in samples]

        max_len = max([len(s) for s in samples])
        batch = []
        for s in samples:
            batch.append(torch.cat([s, torch.ones(max_len - len(s), dtype=torch.long).to(s) * pad_id], dim=0))

        return torch.stack(batch, dim=0)

    def mhop_collate(self, samples, pad_id=0):
        if len(samples) == 0:
            return {}

        batch = {
            'q_input_ids': self.collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'q_mask': self.collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], pad_id),

            'q_sp_input_ids': self.collate_tokens([s["q_sp_codes"]["input_ids"].view(-1) for s in samples], pad_id),
            'q_sp_mask': self.collate_tokens([s["q_sp_codes"]["attention_mask"].view(-1) for s in samples], pad_id),

            'c1_input_ids': self.collate_tokens([s["start_para_codes"]["input_ids"] for s in samples], pad_id),
            'c1_mask': self.collate_tokens([s["start_para_codes"]["attention_mask"] for s in samples], pad_id),

            'c2_input_ids': self.collate_tokens([s["bridge_para_codes"]["input_ids"] for s in samples], pad_id),
            'c2_mask': self.collate_tokens([s["bridge_para_codes"]["attention_mask"] for s in samples], pad_id),

            'neg1_input_ids': self.collate_tokens([s["neg_codes_1"]["input_ids"] for s in samples], pad_id),
            'neg1_mask': self.collate_tokens([s["neg_codes_1"]["attention_mask"] for s in samples], pad_id),

            'neg2_input_ids': self.collate_tokens([s["neg_codes_2"]["input_ids"] for s in samples], pad_id),
            'neg2_mask': self.collate_tokens([s["neg_codes_2"]["attention_mask"] for s in samples], pad_id),

        }

        if "token_type_ids" in samples[0]["q_codes"]:
            batch.update({
                'q_type_ids': self.collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], pad_id),
                'c1_type_ids': self.collate_tokens([s["start_para_codes"]["token_type_ids"] for s in samples], pad_id),
                'c2_type_ids': self.collate_tokens([s["bridge_para_codes"]["token_type_ids"] for s in samples], pad_id),
                "q_sp_type_ids": self.collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples],
                                                     pad_id),
                'neg1_type_ids': self.collate_tokens([s["neg_codes_1"]["token_type_ids"] for s in samples], pad_id),
                'neg2_type_ids': self.collate_tokens([s["neg_codes_2"]["token_type_ids"] for s in samples], pad_id),
            })

        return batch

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
                                    max_length=self.max_q_sp_len, return_tensors="pt", truncation=True)
        q_codes = self.tokenizer(question, max_length=self.max_q_len, return_tensors="pt", truncation=True)

        return {
            "q_codes": q_codes,
            "q_sp_codes": q_sp_codes,
            "start_para_codes": start_para_codes,
            "bridge_para_codes": bridge_para_codes,
            "neg_codes_1": neg_codes_1,
            "neg_codes_2": neg_codes_2,
        }

    def prepare_data(self):

        data_files = {'train': self.train_path, 'dev': self.dev_path}
        if self.do_test:
            data_files.update({'test': self.test_path})

        raw_dataset = load_dataset('json', data_files=data_files)

        train_ds = raw_dataset['train']
        print(("train_ds size before filtering:", len(train_ds)))
        raw_dataset['train'] = train_ds.filter(lambda x: len(x['neg_paras']) > 1)
        print(("train_ds size after filtering low negative samples:", len(raw_dataset['train'])))

        # should we parallelize this here? since it is a pytorch lightning module
        preprocessed_dataset = raw_dataset.map(self.preprocess_data, batched=False, num_proc=self.num_workers)

        # save processed dataset
        preprocessed_dataset.save_to_disk(self.preprocessed_data_dir)

    def setup(self, stage: str):
        dataset = load_from_disk(self.preprocessed_data_dir)
        dataset.set_format('torch', columns=['q_codes', 'q_sp_codes', 'start_para_codes', 'bridge_para_codes',
                                             'neg_codes_1', 'neg_codes_2'], device=self.device)
        if stage == 'fit':
            train_dataset = dataset['train']
            dev_dataset = dataset['dev']

            # todo: check the datacollator we are using
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.mhop_collate
            )
            self.dev_loader = DataLoader(
                dev_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.mhop_collate
            )

        if stage == 'test':
            self.test_loader = DataLoader(
                dataset['test'], batch_size=self.batch_size, shuffle=False, collate_fn=self.mhop_collate
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.dev_loader

    def test_dataloader(self):
        return self.test_loader
