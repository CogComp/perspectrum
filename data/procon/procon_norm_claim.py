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
    data = json.load(json_in)
    reader = csv.DictReader(fin)
    for idx, row in enumerate(reader):
        print(idx)
        old = row["Original_Title"]
        new = row["Normalized_Title"]

        if data[idx]["claim_title"] != old:
            print("Anormally! Expected: {}, Actual: {}".format(data[idx]["claim_title"], old))

        data[idx]["claim_title"] = new

    json.dump(data, fout)

