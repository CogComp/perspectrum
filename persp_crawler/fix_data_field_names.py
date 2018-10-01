import json
import sys
"""
Fixing field names(keys) in claim.perspectives_for and claim.perspectives_against
(title -> perspective_title)
(point -> argument)
(counterpoint -> counter_argument)
(perspective_point -> argument)
(perspective_counterpoint -> counter_argument)

"""


def join_para(description):
    result = []
    for para in description:
        result.append(" ".join(para))

    return result


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python ... [raw_json] [out_path]", file=sys.stderr)
        exit(1)

    raw_json = sys.argv[1]
    out_path = sys.argv[2]

    with open(raw_json) as fin:
        data = json.load(fin)

    for claim in data:
        for ppf in claim["perspectives_for"]:
            ppf["perspective_title"] = ppf.pop("title")
            arg = ppf["argument"] = ppf.pop("point")
            arg["description"] = join_para(arg["description"])
            carg = ppf["counter_argument"] = ppf.pop("counterpoint")
            carg["description"] = join_para(carg["description"])

        for ppf in claim["perspectives_against"]:
            arg = ppf["argument"] = ppf.pop("perspective_point")
            arg["description"] = join_para(arg["description"])
            carg = ppf["counter_argument"] = ppf.pop("perspective_counterpoint")
            carg["description"] = join_para(carg["description"])

    with open(out_path, 'w+') as fout:
        json.dump(data, fout)
