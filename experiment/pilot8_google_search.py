"""
Pilot 8
Use google search to retrieve equivalent perspective titles candidates
Parse results from raw google search
"""
import sys
import json

# Exclude sites that are in our dataset already
EXCLUDE_SITES = ['debatewise', 'idebate', 'procon']


def site_excluded(url):
    for site in EXCLUDE_SITES:
        if site in url:
            return True

    return False


def _clean_snippet(raw_snippet):
    return raw_snippet\
        .replace('\n', ' ')\
        .replace('  ', ' ')


def lstrip_nonalphanum(s):
    if s:
        if s[0].isalnum():
            return s
        else:
            return lstrip_nonalphanum(s[1:])
    else:
        return s

def parse_snippets(raw_snippet):
    raw_snippet = _clean_snippet(raw_snippet)
    pieces = [p.strip() for p in raw_snippet.split('...')]
    pieces = [lstrip_nonalphanum(p) for p in pieces if p.endswith('.')]
    return pieces


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python ... [raw_result] [cand_output]", file=sys.stderr)
        exit(1)

    raw_result = sys.argv[1]
    cand_output = sys.argv[2]

    claim_persp_path = "/home/squirrel/ccg-new/projects/perspective/data/pilot3_twowingos/110718-training-data/claim_perspective.json"

    persps_dict = {}
    with open(claim_persp_path, 'r', encoding='utf-8') as fin:
        persps = json.load(fin)
        for p in persps:
            persps_dict[p['id']] = p['title']

    with open(raw_result, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    out_data = []
    for resp in data:
        persp_id = resp['perspective_id']

        print('Processing: {}'.format(persp_id))
        if 'items' not in resp:
            continue

        items = resp['items']

        # print("Query: {}".format(persps_dict[persp_id]))

        persp_cands = []
        for idx, item in enumerate(items):
            if site_excluded(item['formattedUrl']):
                continue
            cands = parse_snippets(item['snippet'])

            if cands:
                persp_cands.append(cands[0])

        for c in persp_cands:
            out_data.append({
                'perspective_id': persp_id,
                'candidate': c
            })

    with open(cand_output, 'w', encoding='utf-8') as fout:
        json.dump(out_data, fout)
