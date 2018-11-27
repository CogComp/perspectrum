from webapp.models import *
import json


th = 8136
num_original_cands = 4
num_google_cands = 1

def update_persp_candidates():
    """
    Update persp candidates in evidence table
    :return:
    """
    pilot12_result = "/home/squirrel/ccg-new/projects/perspective/data/pilot12_evidence_verification/reverse_persp_candidates.json"
    with open(pilot12_result) as fin:
        pilot12_data = json.load(fin)

    for eid, cands in pilot12_data.items():
        evi = Evidence.objects.get(id=eid)
        p_cands = [c[1] for c in cands]

        origin_p_cands = [c for c in p_cands if c <= th]
        google_p_cands = [c for c in p_cands if c > th]

        # Get gold perspective
        gold_pid = EvidenceRelation.objects.get(evidence_id=eid).perspective_id

        if gold_pid not in origin_p_cands:
            origin_p_cands.insert(0, gold_pid)
        else:
            origin_p_cands.insert(0, origin_p_cands.pop(origin_p_cands.index(gold_pid)))

        evi.origin_candidates = json.dumps(origin_p_cands)
        evi.google_candidates = json.dumps(google_p_cands)
        evi.save()


def generate_evidence_batch(num_evidence_each_bin=10):
    id_list = Evidence.objects.all().exclude(content="").values_list('id', flat=True)

    bin = []
    for id in id_list:
        bin.append(id)
        if len(bin) >= num_evidence_each_bin:
            eb = EvidenceBatch.objects.create(evidence_ids=json.dumps(bin))
            eb.save()
            bin.clear()

    eb = EvidenceBatch.objects.create(evidence_ids=json.dumps(bin))
    eb.save()


if __name__ == '__main__':
    # update_persp_candidates()
    generate_evidence_batch()