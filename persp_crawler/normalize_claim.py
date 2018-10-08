"""
Script to normalize claim (i.e. remove "this house")
Also remove claims without any persps
"""

import sys
import json
import csv


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python ... [raw_json] [fixed_claim] [output_path]", file=sys.stderr)
        exit(1)

    raw_json = sys.argv[1]
    fixed_claim = sys.argv[2]
    output_path = sys.argv[3]

    with open(raw_json, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    fc_list = []
    with open(fixed_claim, 'r', encoding='utf-8') as fin:
        fc = csv.DictReader(fin)
        for line in fc:
            fc_list.append((line["old claim"].rstrip(), line["new claim"].rstrip()))

    index = 0
    count = 0
    unequal = []
    fixed_data = []
    for claim in data:
        claim_pair = fc_list[index]
        # print(claim_pair)
        if claim_pair[1]:
            # if claim doesn't have perspectives, discard it
            if claim["perspectives_for_count"] != 0 or claim["perspectives_against_count"] != 0:
                if claim["claim_title"] != claim_pair[0]:
                    unequal.append((claim["claim_title"], claim_pair[0]))
                claim["claim_title"] = claim_pair[1]
                fixed_data.append(claim)
                count += 1

        index += 1

    print("Total Count: {}".format(count))
    print("Unequak Count: {}".format(len(unequal)))
    print(unequal)
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(fixed_data, fout)
