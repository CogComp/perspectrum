"""
Update perspective equivalent candidates in db

"""

from webapp.models import *
import json
import sys
from nltk import word_tokenize


def validate_perspective(perspective_title):
    toks = word_tokenize(perspective_title)
    return len(toks) > 3


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

    for c in cands:
        pid = c['persp_id']
        cur_p = Perspective.objects.get(id=pid)
        _cands_id = json.loads(cur_p.similar_persps)

        _google_cands = [_c for _c in c['candidates'] if validate_perspective(_c[0])]
        for gc in _google_cands[:5]:
            title = gc[0]
            _q = Perspective.objects.filter(title=title)
            if _q.count() > 0:
                _p = _q.first()
            else:
                _p = Perspective.objects.create(source='google', title=title)
                _p.save()

            cand_pid = _p.id
            if cand_pid not in _cands_id:
                _cands_id.append(cand_pid)

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


def add_persp_rel_google_perspectives(candidates_path):
    with open(candidates_path, 'r') as fin:
        cands = json.load(fin)

    for c in cands:
        pid = c['persp_id']
        cid = PerspectiveRelation.objects.get(perspective_id=pid, author=PerspectiveRelation.GOLD).claim_id

        _google_cands = [_c for _c in c['candidates'] if validate_perspective(_c[0])]
        added_persps = set()
        for gc in _google_cands[:5]:
            title = gc[0]
            _q = Perspective.objects.filter(title=title, source="google")
            if _q.count() == 0:
                continue

            _p = _q.first()
            if _p.id not in added_persps:
                _rel = PerspectiveRelation.objects.create(claim_id=cid, perspective_id=_p.id, author=PerspectiveRelation.GOLD,
                                                        rel="N", comment="google")
                added_persps.add(_p.id)
                _rel.save()


def remove_redundant_gold_persp_rel_annotation():
    count = 0
    for p in PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD):
        if PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=p.claim_id, perspective_id=p.perspective_id).count() > 1:
            p.delete()
            count+=1

    print("Removed {} duplicate records. ".format(count))


def update_persp_token_length():
    for p in Perspective.objects.all():
        if len(word_tokenize(p.title)) < 3:
            print(p.id)
            p.more_than_two_tokens = 0
            p.save()


if __name__ == '__main__':

    # google_candidates = "/home/squirrel/ccg-new/projects/perspective/data/pilot7_persp_equivalence/persp_google_cands.json"
    # lucene_candidates = "/home/squirrel/ccg-new/projects/perspective/data/pilot7_persp_equivalence/persp_lucene_cands.json"

    # google_perspectives_to_db(google_candidates)
    # lucene_perspectives_to_db(lucene_candidates)

    # add_persp_rel_google_perspectives(google_candidates)
    # remove_redundant_gold_persp_rel_annotation()
    update_persp_token_length()