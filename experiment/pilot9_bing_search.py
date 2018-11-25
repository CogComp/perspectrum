"""
Pilot 9
Use Bing to search for equivalent perspectives
"""
import json
import requests
import sys
import json

subscription_key = "e7ef54c352ed46809bb38c6db1017448"
customConfigId = "mysearch"
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

def test():
    search_term = "Countries around the world should intervene in Syria. "

    # url = 'https://api.cognitive.microsoft.com/bingcustomsearch/v7.0/search?q=' + searchTerm + '&customconfig=' + customConfigId
    # r = requests.get(url, headers={'Ocp-Apim-Subscription-Key': subscriptionKey})
    # print(r.text)
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    print(json.dumps(search_results))

def doBingSearch(query):
    params = {"q": query, "textDecorations": False, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    # print(json.dumps(search_results))
    return search_results

def annotateData():
    claim_persp_path = "/Users/daniel/ideaProjects/perspective/data/perspectives.json"

    with open(claim_persp_path, 'r', encoding='utf-8') as fin:
        persps = json.load(fin)
        for i, x in enumerate(persps):
            if i % 10 == 0:
                print("i: " + str(i))
            if x["title"] != None:
                # print(x["title"].strip())
                out = doBingSearch(x["title"].strip())
                persps[i]["bing"] = out
            if i % 300 == 10:
                with open('perspectives_with_bing.json', 'w') as outfile:
                    json.dump(persps, outfile)

# ignore these
EXCLUDE_SITES = ['debatewise', 'idebate', 'procon', 'debatepedia']

def site_excluded(url):
    for site in EXCLUDE_SITES:
        if site in url:
            return True

    return False


def cleanUpBingTitles():
    claim_persp_path = "/Users/daniel/ideaProjects/perspective/experiment/perspectives_with_bing.json"

    with open(claim_persp_path, 'r', encoding='utf-8') as fin:
        persps = json.load(fin)
        for i, x in enumerate(persps):
            print("*" + x["title"])
            # print(json.dumps(x["bing"]))
            for website in x["bing"]["webPages"]["value"]:
                url = website["url"]
                title = website["name"]
                if site_excluded(url):
                    continue
                else:
                    print(" -> " + title)

if __name__ == '__main__':
    cleanUpBingTitles()
    # annotateData()
    # test()
    # if len(sys.argv) != 3:
    #     print("Usage: python ... [raw_result] [cand_output]", file=sys.stderr)
    #     exit(1)
    #
    # raw_result = sys.argv[1]
    # cand_output = sys.argv[2]
