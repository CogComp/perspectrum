import json
import csv

import sys

if len(sys.argv) != 4:
    print("Usage: python ... [norm] [input] [output]", file=sys.stderr)
    exit(1)

norm_claim = sys.argv[1]
input_json = sys.argv[2]
out_json = sys.argv[3]


with open(norm_claim, 'r', encoding='utf-8') as fin, open(input_json, 'r', encoding='utf-8') as json_in, open(out_json, 'w', encoding='utf-8') as fout:
    norm_mapping = {}
    reader = csv.DictReader(fin)
    for row in reader:
        norm_mapping[row["orig"].strip()] = row["normalized"].strip()

    data = json.load(json_in)
    print(len(data))

    drop_list = []
    for idx, c in enumerate(data):
        if c["claim_title"] in norm_mapping:
            normed_claim = norm_mapping[c["claim_title"]]
            if normed_claim:
                c["claim_title"] = normed_claim
            else:
                drop_list.append(idx)

    for i in sorted(drop_list, reverse=True):
        del data[i]

    print(len(data))

    json.dump(data, fout)

