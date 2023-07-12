import json
import argparse
from collections import defaultdict
import random


def extract_evidence_relations_pairs(docred_example, rel_info_dict):

    sents = docred_example['sents']
    evidences = []
    for sent in sents:
        if "." in sent:
            sent.remove(".")
        text = " ".join(sent)
        text = text.strip()
        text = text.replace(" . ", ". ")
        text = text.replace(" , ", ", ")
        evidences.append(text)

    entity_sets = docred_example['vertexSet']
    entity_list = []

    for entity_set in entity_sets:
        entity_list.append(entity_set[0]['name'])

    relations = docred_example['labels']
    evidence_rels = defaultdict(list)

    for rel in relations:
        if len(rel['evidence']) == 0:
            continue
        evidence_txt = " ".join([evidences[e] for e in rel['evidence']])
        if evidence_txt in evidence_rels:
            evidence_rels[evidence_txt].append(f"{entity_list[rel['h']]} | {rel_info_dict[rel['r']]} | {entity_list[rel['t']]}")
        else:
            evidence_rels[evidence_txt] = [f"{entity_list[rel['h']]} | {rel_info_dict[rel['r']]} | {entity_list[rel['t']]}"]

    seq2seq_data = []

    for evidence, rels in evidence_rels.items():
        seq2seq_data.append({
            "input": evidence,
            "output": "\n".join(rels),
        })

    return seq2seq_data


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data_path", type=str, default="data/Re-DocRED-main/data/train_revised.json")
    argparse.add_argument("--rel_info_path", type=str, default="data/Re-DocRED-main/rel_info.json")
    args = argparse.parse_args()

    data_path = args.data_path
    rel_info_path = args.rel_info_path

    with open(data_path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

    with open(rel_info_path, 'r') as f:
        rel_info = f.read()
        rel_info_dict = json.loads(rel_info)

    processed_data = []
    for data_ex in data_json:
        evidence_relations = extract_evidence_relations_pairs(data_ex, rel_info_dict)
        for er in evidence_relations:
            processed_data.append(er)

    print("Number of examples: ", len(processed_data))

    with open("data/Re-DocRED-main/evidence_rels.json", 'w') as f:
        for data_ex in processed_data:
            f.write(f"{json.dumps(data_ex)}\n")



