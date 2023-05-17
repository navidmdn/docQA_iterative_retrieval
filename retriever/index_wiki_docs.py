import argparse

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoConfig
from retriever.roberta_retriever import RobertaRetriever
from typing import List, Dict
import datasets


class DocIndexer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def batch_encode(self, docs: Dict) -> Dict:
        """
        Encode a batch of documents into vectors.
        """
        titles = [d.strip() for d in docs['title']]
        texts = [d.strip() for d in docs['text']]

        # encode titles and texts
        docs_tokenized = self.tokenizer(titles, text_pair=texts, truncation=True, max_length=512, padding='longest',
                                        return_tensors='pt')

        with torch.no_grad():
            docs_encoded = self.model.encode_seq(docs_tokenized['input_ids'], docs_tokenized['attention_mask'])

        return {'doc_emb': docs_encoded}


def main():
    # parse args and get model and tokenizer and path to documents
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--encoder_path", type=str, default="results/checkpoint-1/pytorch_model.bin")
    parser.add_argument("--data_path", type=str, default="data/hotpot_index/test_id2doc.json")
    parser.add_argument("--index_path", type=str, default="data/hotpot_index/wiki_index.json")
    parser.add_argument("--cache_dir", type=str, default="models/huggingface")

    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = RobertaRetriever(config, args.model_name_or_path, args.cache_dir)
    model.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaRetriever(config, args.model_name_or_path, args.cache_dir)
    model.load_state_dict(torch.load(args.encoder_path))

    # load documents
    indexer = DocIndexer(model, tokenizer)
    docs_ds = datasets.load_dataset('json', data_files=args.data_path)
    docs_ds = docs_ds.map(indexer.batch_encode, batched=True, batch_size=32, remove_columns=['title', 'text'])
    print(len(docs_ds))


if __name__ == "__main__":
    main()