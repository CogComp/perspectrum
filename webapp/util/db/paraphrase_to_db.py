# Import paraphrase into perspective table

from webapp.models import PerspectiveParaphrase, Perspective, EquivalenceBatch
from nltk import word_tokenize
import json


def clean_similar_persps():
    for p in Perspective.objects.all().exclude(similar_persps="[]"):
        p.similar_persps = "[]"
        p.save()


def import_paraphrase():
    q = PerspectiveParaphrase.objects.all()

    for pp in q:
        pid = pp.perspective_id
        gold_p = Perspective.objects.get(id=pid)

        cands = json.loads(pp.user_generated)

        # Only keep those with larger or equal to 3 tokens
        cands = [c for c in cands if len(word_tokenize(c)) > 3]

        new_pids = []
        for c in cands:
            p = Perspective.objects.create(source="paraphrase", title=c, similar_persps=json.dumps([pid]))
            new_pids.append(p.id)
            p.save()

        if len(new_pids) > 0:
            gold_p.similar_persps = json.loads(gold_p.similar_persps) + new_pids
            gold_p.save()


def import_potentially_equivalent_perspectives(candidates=2):
    path = "data/pilot17_making_the_dataset/persps_of_claims_with_gt10_persps.json"

    with open(path) as fin:
        data = json.load(fin)

    for d in data:
        pid = d["perspective_id"]
        p = Perspective.objects.get(id=pid)
        cands = [k for k in d["candidates"] if k != pid]
        p.similar_persps = json.loads(p.similar_persps) + cands[:candidates]
        p.save()


def import_equivalence_batches(batch_size=8):
    bin = []
    q = Perspective.objects.filter(source__in=["idebate", "debatewise", "procon"]).exclude(similar_persps='[]')
    print(len(q))

    for p in q:
        bin.append(p.id)
        if len(bin) >= batch_size:
            eb = EquivalenceBatch.objects.create(perspective_ids=json.dumps(bin))
            bin.clear()
            eb.save()


if __name__ == '__main__':

    # clean_similar_persps()
    # import_paraphrase()
    # import_potentially_equivalent_perspectives()
    import_equivalence_batches()
