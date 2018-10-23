import json
import csv

import sys

if len(sys.argv) != 4:
    print("Usage: python ... [norm] [input] [output]", file=sys.stderr)
    exit(1)

norm_claim = sys.argv[1]
input_json = sys.argv[2]
out_json = sys.argv[3]


with open(norm_claim, 'r') as fin, open(input_json, 'r') as json_in, open(out_json, 'w') as fout:
    norm_mapping = {}
    reader = csv.DictReader(fin)
    for row in reader:
        norm_mapping[row["orig"].strip()] = row["normalized"].strip()

    data = json.load(json_in)

    for c in data:
        if c["claim_title"] in norm_mapping:
            c["claim_title"] = norm_mapping[c["claim_title"]]

    json.dump(data, fout)

