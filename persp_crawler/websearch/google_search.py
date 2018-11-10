"""
Google custom search API
"""
import requests

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

    if len(sys.argv) != 3:
        print("Usage: python ... [api_key] [cx]", file=sys.stderr)
        exit(1)

    api_key = sys.argv[1]
    cx = sys.argv[2]
    gs = GoogleSearch(api_key, cx)
    res = gs.search('We Should Have a Quota for Women on Corporate Boards. Putting a Quota in Place Would Help Women')
    print(res)
