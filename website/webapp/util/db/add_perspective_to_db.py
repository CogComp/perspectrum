"""
Update perspective equivalent candidates in db

"""

from webapp.models import *
import json
import sys

def update_perspectives_in_db(persps, source):
    """

    :param persps: an iterable of perspective titles (assume data is clean)
    :param source: source of the perspective, for example 'idebate', 'google'
    :return:
    """
    if type(persps) != set:
        persps = set(persps)

    for p in persps:
        pp = Perspective.objects.create(source=source, title=p)
        pp.save()

def google_perspectives_to_db(candidates_path):
    """

    :param candidates_path: Produced in pilot 7
    :return:
    """
    with open(candidates_path, 'r') as fin:
        cands = json.load(fin)

    candidate_set = set()
    for c in cands:
        pid = c['persp_id']
        cur_p = Perspective.objects.get(id=pid)
        _cands_id = json.loads(cur_p.similar_persps)
        for gc in c['candidates'][:5]:
            title = gc[0]
            candidate_set.add(title)
            _p = Perspective.objects.create(source='google', title=title)
            _p.save()
            _cands_id.append(_p.id)

        cur_p.similar_persps = _cands_id
        cur_p.save()

def lucene_perspectives_to_db(candidates_path):
    """

    :param candidates_path:
    :return:
    """

    with open(candidates_path, 'r') as fin:
        cands = json.load(fin)

    for c in cands:
        pid = c['persp_id']
        _cands_id = [p[0] for p in c['candidates'][:6] if p[0] != pid]
        _cands_id = _cands_id[:5]

        cur_p = Perspective.objects.get(id=pid)
        _cands_id = json.loads(cur_p.similar_persps) + _cands_id
        cur_p.similar_persps = _cands_id
        cur_p.save()


def add_persp_rel_google_perspectives():
    q = Perspective.objects.all().exclude(similar_persps="[]")
    for p in q:
        pid = p.id
        cid = PerspectiveRelation.objects.get(perspective_id=pid, author=PerspectiveRelation.GOLD).claim_id
        google_pids = json.loads(p.similar_persps)[:5]

        for gpid in google_pids:
            _p = PerspectiveRelation.objects.create(claim_id=cid, perspective_id=gpid, author=PerspectiveRelation.GOLD,
                                               rel="N", comment="google")
            _p.save()

if __name__ == '__main__':

    # google_candidates = "/home/squirrel/ccg-new/projects/perspective/data/pilot7_persp_equivalence/persp_google_cands.json"
    # lucene_candidates = "/home/squirrel/ccg-new/projects/perspective/data/pilot7_persp_equivalence/persp_lucene_cands.json"

    # google_perspectives_to_db(google_candidates)
    # lucene_perspectives_to_db(lucene_candidates)

    add_persp_rel_google_perspectives()
