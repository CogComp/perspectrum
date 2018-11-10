"""
Google custom search API
"""
import requests
import json
import time

API_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleSearch:
    def __init__(self, api_key, cx):
        """
        https://developers.google.com/custom-search/v1/using_rest
        :param api_key: google cse api key
        :param cx: custom search engine key
        """

        self.api_key = api_key
        self.cx = cx

    def search(self, q, start_index=None):
        payload = {
            'key': self.api_key,
            'cx': self.cx,
            'q': q,
        }
        if start_index:
            payload['start'] = start_index

        return requests.get(API_URL, params=payload).json()


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        print("Usage: python ... [api_key] [cx] [claim_persp_json] [out_path]", file=sys.stderr)
        exit(1)

    api_key = sys.argv[1]
    cx = sys.argv[2]
    claim_persp_path = sys.argv[3]
    out_path = sys.argv[4]

    gs = GoogleSearch(api_key, cx)

    # Get all valid repsonses from before
    with open(out_path, 'r', encoding='utf-8') as fin:
        resps = json.load(fin)
        resps = [p for p in resps if "error" not in p]
        processed_pids = {res['perspective_id'] for res in resps}

    with open(claim_persp_path, 'r', encoding='utf-8') as fin:
        cps = json.load(fin)

    for cp in cps:
        id = cp['id']
        if id not in processed_pids:
            print('Searching perspective id = {}'.format(id))
            res = gs.search(cp['title'], start_index=2)
            res['perspective_id'] = id
            resps.append(res)
            time.sleep(0.8)

    with open(out_path, 'w') as fout:
        json.dump(resps, fout)
