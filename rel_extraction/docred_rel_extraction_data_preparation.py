import json
import argparse


def extract_passage_relations(docred_example, rel_info_dict):
    sents = docred_example['sents']
    text = ""
    for sent in sents:
        text += " ".join(sent) + " "
    text = text.strip()
    text = text.replace(" . ", ". ")
    text = text.replace(" , ", ", ")

    entity_sets = docred_example['vertexSet']
    entity_list = []

    for entity_set in entity_sets:
        entity_list.append(entity_set[0]['name'])

    relations = docred_example['labels']
    relation_list = []

    for rel in relations:
        relation_list.append((entity_list[rel['h']], rel_info_dict[rel['r']], entity_list[rel['t']]))

    relations_txt = "\n".join([f"{rel[0]} | {rel[1]} | {rel[2]}" for rel in relation_list])

    return text, relations_txt


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data_path", type=str, default="data/Re-DocRED-main/data/dev_revised.json")
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
        text, relation_text = extract_passage_relations(data_ex, rel_info_dict)
        processed_data.append({'paragraph': text, 'relations': relation_text})

    with open("../data/Re-DocRED-main/docred_pr_pairs_dev.json", 'w') as f:
        for data_ex in processed_data:
            f.write(f"{json.dumps(data_ex)}\n")



