from webapp.models import Claim
import json

KEYWORDS_NUM = 4 # Numbers of keywords wanted for each claim

if __name__ == '__main__':
    keywords_file = "data/pilot13_keyword_extraction/claims_with_topics.json"

    with open(keywords_file) as fin:
        data = json.load(fin)

    for _c in data:
        cid = _c["claim_id"]
        c = Claim.objects.get(id=cid)
        keywords = _c["noun_phrases"][:KEYWORDS_NUM]
        keywords = [k[0] for k in keywords]
        print(c)
        c.keywords = json.dumps(keywords)
        c.save()
