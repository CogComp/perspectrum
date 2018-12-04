# Import paraphrase into perspective table

from webapp.models import PerspectiveParaphrase, Perspective, EquivalenceBatch, ReStep1Results
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

    # Find valid
    _rq = ReStep1Results.objects.filter(p_i_3__gt =0.5, label_3__in=["S", "U"])
    q = Perspective.objects.filter(source__in=["idebate", "debatewise", "procon"]).exclude(similar_persps='[]')
    print(len(q))

    for p in q:
        cids = list(_rq.filter(perspective_id=p.id).values_list('claim_id', flat=True))
        for cid in cids:
            bin.append((cid, p.id))
            if len(bin) >= batch_size:
                eb = EquivalenceBatch.objects.create(perspective_ids=json.dumps(bin))
                bin.clear()
                eb.save()


def remove_trivial_paraphrases():
    q = Perspective.objects.filter(source='paraphrase')
    for pp in q:
        root_pod = json.loads(pp.similar_persps)[0]
        rp = Perspective.objects.get(id=root_pod)

        if rp.title.strip() == pp.title.strip():
            print("Trivial paraphrase detected! persp id = {}, paraphrase id = {}".format(root_pod, pp.id))
            rp.similar_persp = json.dumps([_p for _p in json.loads(rp.similar_persp) if _p == pp.id])
            rp.save()
            pp.delete()


if __name__ == '__main__':

    # clean_similar_persps()
    # import_paraphrase()
    # import_potentially_equivalent_perspectives()
    # import_equivalence_batches()
    remove_trivial_paraphrases()