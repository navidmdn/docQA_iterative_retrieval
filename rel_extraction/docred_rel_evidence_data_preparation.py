import json
import argparse
from collections import defaultdict
import random


def extract_evidence_relation_pairs(docred_example, rel_info_dict, n_neg=1, no_relation_token="<NONE>"):

    all_rels = list(rel_info_dict.values())
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
    result_paris = []

    head_rels = defaultdict(list)

    # generating positive pairs
    for rel in relations:
        head_rels[rel['h']].append(rel['r'])

        if len(rel['evidence']) == 0:
            continue
        result_paris.append({
            'head': entity_list[rel['h']],
            'relation': rel_info_dict[rel['r']],
            'tail': entity_list[rel['t']],
            'evidence': " ".join([evidences[e] for e in rel['evidence']])
        })

    # generating negative pairs
    for _ in range(n_neg):
        for head in head_rels.keys():
            random_rel = random.choice(all_rels)
            while random_rel in head_rels[head]:
                random_rel = random.choice(all_rels)

            # to avoid overlapping negative pairs
            head_rels[head].append(random_rel)

            # select a random hard evidence
            related_evidences = [sent for sent in evidences if entity_list[head] in sent]
            if len(related_evidences) == 0:
                continue

            random_evidence = random.choice(related_evidences)

            result_paris.append({
                'head': entity_list[head],
                'relation': random_rel,
                'tail': no_relation_token,
                'evidence': random_evidence
            })

    return result_paris


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
        evidence_relations = extract_evidence_relation_pairs(data_ex, rel_info_dict)
        for evidence_relation in evidence_relations:
            processed_data.append({
                "input": f"{evidence_relation['evidence']}<head>{evidence_relation['head']}<rel>{evidence_relation['relation']}<tail>",
                "output": evidence_relation['tail'],
                # just to conform to the format of the original data
                "instruction": ""
            })

    print("Number of examples: ", len(processed_data))

    with open("data/Re-DocRED-main/rel_evidence.json", 'w') as f:
        for data_ex in processed_data:
            f.write(f"{json.dumps(data_ex)}\n")



