

import argparse
import collections
import json
import logging
import datasets
import time
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from retriever.roberta_retriever import RobertaRetriever
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def load_model_and_tokenizer(base_model_name, encoder_path, device, cache_dir=None):
    config = AutoConfig.from_pretrained(base_model_name)
    model = RobertaRetriever(config, base_model_name, cache_dir)

    state_dict_path = os.path.join(encoder_path, 'pytorch_model.bin')
    print("working on device: {}".format(device))

    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, cache_dir=cache_dir)
    model.eval()

    return model, tokenizer


def encode_batch_sentence_pairs(model, tokenizer, sent1s, sent2s=None, device=torch.device('cpu'), max_len=512):
    inputs = tokenizer(sent1s, text_pair=sent2s, max_length=max_len, return_tensors="pt", truncation=True,
                       padding='longest').to(device)

    with torch.no_grad():
        outputs = model.encode_seq(input_ids=inputs['input_ids'], mask=inputs['attention_mask'])

    return outputs.cpu().numpy()

def evaluate(model, tokenizer, id2doc, args):
    pass

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            data.append(record)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, default=None)
    parser.add_argument('--indexpath', type=str, default=None)
    parser.add_argument('--corpus_ds', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='models/huggingface')
    parser.add_argument('--base_model_name', type=str, default='roberta-base')
    parser.add_argument('--encoder_path', type=str, default='results/checkpoint-1')
    parser.add_argument('--topk', type=int, default=2, help="topk paths")

    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--max-q-len', type=int, default=70)
    parser.add_argument('--max-c-len', type=int, default=300)
    parser.add_argument('--max-q-sp-len', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--model-name', type=str, default='roberta-base')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--save-index', action="store_true")
    parser.add_argument('--only-eval-ans', action="store_true")
    parser.add_argument('--shared-encoder', action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument('--hnsw', action="store_true")
    args = parser.parse_args()

    dataset_jsonl = load_jsonl(args.raw_data)
    logger.info(f"Corpus size {len(dataset_jsonl)}")

    indexed_corpus = datasets.load_dataset('json', data_files=args.corpus_ds)["train"]
    indexed_corpus.load_faiss_index('embeddings', args.indexpath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_and_tokenizer(args.base_model_name, args.encoder_path, device=device,
                                                cache_dir=args.cache_dir)
    evaluate(model, tokenizer, dataset_jsonl, args)

    logger.info("Encoding questions and searching")
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in dataset_jsonl]
    metrics = []
    retrieval_outputs = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ann = dataset_jsonl[b_start:b_start + args.batch_size]
            bsize = len(batch_q)

            q_embs = encode_batch_sentence_pairs(model, tokenizer, batch_q, device=device, max_len=args.max_q_len)
            scores, docs = indexed_corpus.get_nearest_examples_batch('embeddings', q_embs, k=args.topk)

            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for i, _ in enumerate(docs[b_idx]["id"]):
                    doc_text = docs[b_idx]["text"][i]
                    if doc_text.strip() == "":
                        scores[b_idx][i] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc_text))

            q_sp_embs = encode_batch_sentence_pairs(model, tokenizer, sent1s=[_[0] for _ in query_pairs],
                                                    sent2s=[_[1] for _ in query_pairs], device=device,
                                                    max_len=args.max_q_sp_len)

            scores2h, docs2h = indexed_corpus.get_nearest_examples_batch('embeddings', q_sp_embs, k=args.topk)

            scores = np.array(scores)
            scores2h = np.array(scores2h)

            scores2h = scores2h.reshape(bsize, args.topk, args.topk)
#            docs2h = docs2h.reshape(bsize, args.topk, args.topk)

            # aggregate scores: multiply or add?
            path_scores = np.expand_dims(scores, axis=2) + scores2h

            for idx in range(bsize):
                search_scores = path_scores[idx]
                # ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],
                #                                           (args.beam_size, args.beam_size))).transpose()
                ranked_pairs = np.argsort(search_scores.ravel())[::-1]
                retrieved_titles = []
                hop1_titles = []
                paths, path_titles = [], []

                for i in range(args.topk):
                    path_id = ranked_pairs[i]
                    hop1idx = path_id // args.topk
                    hop2idx = path_id % args.topk

                    hop_1_doc = {k: docs[idx][k][hop1idx] for k in docs[idx]}
                    hop_2_doc = {k: docs2h[idx*args.topk + hop1idx][k][hop2idx] for k in docs2h[idx*args.topk + hop1idx]}
                    retrieved_titles.append(hop_1_doc["title"])
                    retrieved_titles.append(hop_2_doc["title"])

                    paths.append([hop_1_doc, hop_2_doc])
                    path_titles.append([hop_1_doc["title"], hop_2_doc["title"]])
                    hop1_titles.append(hop_1_doc["title"])

                sp = batch_ann[idx]["sp"]
                assert len(set(sp)) == 2
                type_ = batch_ann[idx]["type"]
                question = batch_ann[idx]["question"]
                p_recall, p_em = 0, 0
                sp_covered = [sp_title in retrieved_titles for sp_title in sp]
                if np.sum(sp_covered) > 0:
                    p_recall = 1
                if np.sum(sp_covered) == len(sp_covered):
                    p_em = 1
                path_covered = [int(set(p) == set(sp)) for p in path_titles]
                path_covered = np.sum(path_covered) > 0
                recall_1 = 0
                covered_1 = [sp_title in hop1_titles for sp_title in sp]
                if np.sum(covered_1) > 0: recall_1 = 1
                metrics.append({
                    "question": question,
                    "p_recall": p_recall,
                    "p_em": p_em,
                    "type": type_,
                    'recall_1': recall_1,
                    'path_covered': int(path_covered)
                    })

                retrieval_outputs.append({
                    "_id": batch_ann[idx]["_id"],
                    "question": batch_ann[idx]["question"],
                    "candidate_chains": paths,
                    # "sp": sp_chain,
                    # "answer": gold_answers,
                    # "type": type_,
                    # "coverd_k": covered_k
                })

    if args.save_path != "":
        with open(args.save_path, "w") as out:
            for l in retrieval_outputs:
                out.write(json.dumps(l) + "\n")

    logger.info(f"Evaluating {len(metrics)} samples...")
    type2items = collections.defaultdict(list)
    for item in metrics:
        type2items[item["type"]].append(item)
    if args.only_eval_ans:
        logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
    else:
        logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in metrics])}')
        logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in metrics])}')
        logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in metrics])}')
        logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in type2items[t]])}')
            logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in type2items[t]])}')
            logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in type2items[t]])}')
            logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in type2items[t]])}')
