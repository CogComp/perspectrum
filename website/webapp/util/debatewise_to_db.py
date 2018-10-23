from webapp.models import *
from webapp.views import load_json
import sys

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: python ... [json_path]", file=sys.stderr)
    #     exit(1)

    # json_path = sys.argv[1]
    json_path = "/mnt/e/work/cogcomp-new/projects/perspective/data/debatewise/debatewise.json"
    data = load_json(json_path)

    SOURCE = "debatewise"

    for _c in data:
        title = _c["claim_title"]
        claim = Claim.objects.create(source=SOURCE, title=title)
        claim.save()

        for _p in _c["perspectives_for"]:
            p_title = _p["perspective_title"]
            evidence = " ".join(_p["argument"]["description"])
            persp = Perspective.objects.create(source=SOURCE, title=p_title)
            persp_rel = PerspectiveRelation.objects.create(author=PerspectiveRelation.GOLD,
                                                    claim_id=claim.id, perspective_id=persp.id, rel='S')
            persp.save()
            persp_rel.save()

            if evidence:
                evi = Evidence.objects.create(source=SOURCE, content=evidence)
                ev_rel = EvidenceRelation.objects.create(author=EvidenceRelation.GOLD,
                                                         perspective_id=persp.id, evidence_id=evi.id)
                ev_rel.save()

        for _p in _c["perspectives_against"]:
            p_title = _p["perspective_title"]
            evidence = " ".join(_p["argument"]["description"])
            persp = Perspective.objects.create(source=SOURCE, title=p_title)
            persp_rel = PerspectiveRelation.objects.create(author=PerspectiveRelation.GOLD,
                                                    claim_id=claim.id, perspective_id=persp.id, rel='U')
            persp.save()
            persp_rel.save()

            if evidence:
                evi = Evidence.objects.create(source=SOURCE, content=evidence)
                ev_rel = EvidenceRelation.objects.create(author=EvidenceRelation.GOLD,
                                                         perspective_id=persp.id, evidence_id=evi.id)
                ev_rel.save()

