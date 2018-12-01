import json
from webapp.models import *

if __name__ == '__main__':

    json_path = "data/relevant_senetences_to_perspectives/relevant_sentences_to_perspectives.json"

    with open(json_path) as fin:
        hints = json.load(fin)

    valid_ids = set(Perspective.objects.filter(pilot1_have_stance=True).values_list("id", flat=True).distinct())

    for p in hints:
        pid = p["id"]
        if pid in valid_ids:
            for cand in p["sentences"]:
                content = cand[0]
                score = cand[1]
                p = PerspectiveParaphraseHints.objects.create(perspective_id=pid, content=content, lucene_score=score)
                p.save()
