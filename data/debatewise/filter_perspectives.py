"""
Filter out low-quality perspectives, specifically those that--
    1. are too short
    2. (in debatewise) too closely resembles the start of evidence
"""
import json
import os
import nltk
import sys

def title_length_filter(title_toks):
    return len(title_toks) > 3


def evidence_overlap_filter(title_toks, evidence_toks):
    title_toks_set = set(title_toks)
    evidence_toks = evidence_toks[:len(title_toks)]
    evidence_toks_set = set(evidence_toks)
    intersect = title_toks_set & evidence_toks_set

    return (len(intersect) / len(title_toks_set)) < 0.75


def valid_persp(persp):
    """
    True if perspective title length is larger than
    :return:
    """
    title = persp["perspective_title"]
    evidence = " ".join(persp["argument"]["description"])

    title_toks = nltk.word_tokenize(title)

    evidence_toks = nltk.word_tokenize(evidence)

    return title_length_filter(title_toks) & evidence_overlap_filter(title_toks, evidence_toks)


def clean_persps(persps):
    drop_list = []
    for idx, p in enumerate(persps):
        if not valid_persp(p):
            drop_list.append(idx)

    for index in sorted(drop_list, reverse=True):
        del persps[index]

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python ... [orig_json] [new_json]", file=sys.stderr)
        exit(1)

    orig_json = sys.argv[1]
    new_json = sys.argv[2]

    with open(orig_json, 'r') as fin:
        data = json.load(fin)

    drop_claim_list = []
    idx = 0
    persp_for_count = 0
    persp_against_count = 0

    for claim in data:
        print("Claim {}: {}".format(idx, claim["claim_title"]))
        print("Before -- \t For count = {} \t Against count = {}".format(len(claim["perspectives_for"]), len(claim["perspectives_against"])))
        clean_persps(claim["perspectives_for"])
        clean_persps(claim["perspectives_against"])
        claim["perspectives_for_count"] = len(claim["perspectives_for"])
        claim["perspectives_against_count"] = len(claim["perspectives_against"])
        if claim["perspectives_for_count"] == 0 and claim["perspectives_against_count"] == 0:
            drop_claim_list.append(idx)

        persp_for_count += claim["perspectives_for_count"]
        persp_against_count += claim["perspectives_against_count"]
        idx += 1
        print("After -- \t For count = {} \t Against count = {}".format(len(claim["perspectives_for"]), len(claim["perspectives_against"])))

    for i in sorted(drop_claim_list, reverse=True):
        del data[i]

    print(len(data))

    with open(new_json, 'w') as fout:
        json.dump(data, fout)


