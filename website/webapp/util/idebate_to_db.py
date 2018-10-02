from webapp.models import *
from webapp.views import load_json
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python ... [json_path]", file=sys.stderr)
        exit(1)

    json_path = sys.argv[1]
    data = load_json(json_path)

    SOURCE = "idebate"

    for _c in data:
        title = _c["claim_title"]
        claim = Claim.objects.create(source=SOURCE, title=title)
        claim.save()

        for _p in _c["perspectives_for"]:
            p_title = _p["perspective_title"]
            evidence = " ".join(_p["argument"]["description"])
            persp = Perspective.objects.create(source=SOURCE, title=p_title, evidence=evidence)
            rel = RelationAnnotation.objects.create(author=RelationAnnotation.GOLD,
                                                    claim_id=claim.id, perspective_id=persp.id, rel='S')
            persp.save()
            rel.save()

        for _p in _c["perspectives_against"]:
            p_title = _p["perspective_title"]
            evidence = " ".join(_p["argument"]["description"])
            persp = Perspective.objects.create(source=SOURCE, title=p_title, evidence=evidence)
            rel = RelationAnnotation.objects.create(author=RelationAnnotation.GOLD,
                                                    claim_id=claim.id, perspective_id=persp.id, rel='O')
            persp.save()
            rel.save()

