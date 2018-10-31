import sys
import os
import json
import pandas as pd


def write_claims_json(claims_list, path):
    with open(path, 'w') as fout:
        fout.write('[')
        for c in claims_list:
            json.dump(c, fout)
            fout.write(',\n')
        fout.write(']')


def write_claims_csv(claims_list, path):
    df = pd.DataFrame(claims_list)
    df.to_csv(path, index=False)


def fixed_csv_to_json(csv_path, good_path, out_path):
    df = pd.read_csv(csv_path)

    claims = []
    for idx, row in df.iterrows():
        row_list = [t for t in row.tolist() if type(t) == str]
        print(row_list)
        claim_title = row_list.pop(0)
        url = row_list.pop(0)

        persp_count = int(len(row_list) / 2)

        persp_for = []
        persp_against = []

        for i in range(persp_count):
            persp_title = row_list[2 * i]
            persp_evi = row_list[2 * i + 1]

            evidence = persp_evi.split("\n")
            evidence = [p for p in evidence if len(p) > 5]  # Filter out stuff like \n or spaces

            if persp_title.startswith("[PRO]"):
                persp_title = persp_title[5:]
                persp_for.append({
                    "perspective_title": persp_title,
                    "argument": {
                        "description": evidence,
                        "reference": []
                    },
                    "counter_argument": []
                })
            elif persp_title.startswith("[CON]"):
                persp_title = persp_title[5:]
                persp_against.append({
                    "perspective_title": persp_title,
                    "argument": {
                        "description": evidence,
                        "reference": []
                    },
                    "counter_argument": []
                })

        claims.append({
            "url": url,
            "claim_title": claim_title,
            "topic": [],
            "perspectives_for_count": len(persp_for),
            "perspectives_for": persp_for,
            "perspectives_against_count": len(persp_against),
            "perspectives_against": persp_against,
        })

    # Load good ones from file
    with open(good_path, 'r') as fin,  open(out_path, 'w') as fout:
        good_claims = json.loads(fin.read())

        merged = good_claims + claims
        json.dump(merged, fout)


# Convert fixed csv to json
if __name__ == '__main__':
    if len(sys.argv) != 2:
            print("Usage: python ... [dir]", file=sys.stderr)
            exit(1)

    base_dir = sys.argv[1]
    fixed_path = os.path.join(base_dir, "procon_empty.csv")
    good_path = os.path.join(base_dir, "procon_good.json")
    merged_path = os.path.join(base_dir, "procon.json")

    fixed_csv_to_json(fixed_path, good_path, merged_path)

# if __name__ == '__main__':
#
#     if len(sys.argv) != 3:
#         print("Usage: python ... [orig_json] [out_json_dir]", file=sys.stderr)
#         exit(1)
#
#     orig_json = sys.argv[1]
#     out_json = sys.argv[2]
#
#     good_claims = []
#     no_persp = []
#     empty_claims = []
#
#     with open(orig_json, 'r') as fin:
#         data = json.load(fin)
#         for claim in data:
#             claim["topic"] = ""
#             if not claim["claim_title"]:
#                 empty_claims.append(claim)
#             elif claim["perspectives_for_count"] == 0:
#                 no_persp.append(claim)
#             else:
#                 good_claims.append(claim)
#
#     write_claims_json(good_claims, os.path.join(out_json, "procon_good.json"))
#     write_claims_csv(empty_claims, os.path.join(out_json, "procon_empty.json"))
#     write_claims_csv(no_persp, os.path.join(out_json, "procon_only_title.json"))

