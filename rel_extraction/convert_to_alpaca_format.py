import json


def run(data_path):

    data = []

    with open(data_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)

    postprocessed_data = []
    for obj in data:
            postprocessed_data.append({
                "instruction": "### paragraph:",
                "input": obj["paragraph"],
                "output": obj["relations"],
            })

    with open(f"{data_path.replace('.json', '-alpaca')}.json", 'w') as f:
        f.write(json.dumps(postprocessed_data))


if __name__ == '__main__':
    file_path = '../data/Re-DocRED-main/docred_pr_pairs_train.json'
    run(file_path)