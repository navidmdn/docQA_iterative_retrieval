import argparse
import json
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoConfig
from retriever.roberta_retriever import RobertaRetriever
from typing import List, Dict
import os
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        titles = [d.strip() for d in docs['title']]
        texts = [d.strip() for d in docs['text']]

        # encode titles and texts
        docs_tokenized = self.tokenizer(titles, text_pair=texts, truncation=True, max_length=512, padding='longest',
                                        return_tensors='pt')

        with torch.no_grad():
            input_ids = docs_tokenized['input_ids'].to(device)
            masks = docs_tokenized['attention_mask'].to(device)
            docs_encoded = self.model.encode_seq(input_ids, masks).cpu()

        return {'doc_emb': docs_encoded}


def main():
    # parse args and get model and tokenizer and path to documents
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--encoder_path", type=str, default="results/checkpoint-1")
    parser.add_argument("--data_path", type=str, default="data/hotpot_index/wiki_id2doc.jsonl")
    parser.add_argument("--index_path", type=str, default="data/hotpot_index/wiki_index.faiss")
    parser.add_argument("--cache_dir", type=str, default="models/huggingface")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if args.data_path.endswith('.json'):
        print("converting json to jsonl ...")

        with open(args.data_path, 'r') as f:
            content = f.read().encode('utf-8')
            data = json.loads(content)

        jsonl_path = args.data_path.replace('.json', '.jsonl')
        with open(jsonl_path, 'w') as fw:
            for k, v in data.items():
                fw.write(json.dumps({"id": k, **v}) + '\n')
    else:
        jsonl_path = args.data_path

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = RobertaRetriever(config, args.model_name_or_path, args.cache_dir)

    state_dict_path = os.path.join(args.encoder_path, 'pytorch_model.bin')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("working on device: {}".format(device))

    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, cache_dir=args.cache_dir)

    # load documents
    indexer = DocIndexer(model, tokenizer)
    docs_ds = datasets.load_dataset('json', data_files=jsonl_path, cache_dir=args.cache_dir)['train']
    docs_ds = docs_ds.map(indexer.batch_encode, batched=True, batch_size=args.batch_size,
                          remove_columns=['title', 'text', 'sents'], num_proc=args.num_proc)
    docs_ds.add_faiss_index(column='doc_emb')

    # save index
    docs_ds.save_faiss_index('doc_emb', args.index_path)


if __name__ == "__main__":
    main()