import json
from webapp.models import *


def update_paraphrase_table(data_path):
    """
    Import lucene data into paraphrase table
    :param data_path: lucene output hints path
    :return:
    """

    valid_ids = set(Perspective.objects.filter(pilot1_have_stance=True, more_than_two_tokens=True).values_list("id", flat=True).distinct())

    with open(data_path) as fin:
        hints = json.load(fin)

    for p in hints:
        pid = p["id"]
        if pid in valid_ids:
            hints = [h[0].strip() for h in p["sentences"]][:15] # keep 15 hints per perspective

            p = PerspectiveParaphrase.objects.create(perspective_id=pid, hints=json.dumps(hints))
            p.save()


def generate_paraphrase_batches(batch_size=5):
    id_list = PerspectiveParaphrase.objects.all().values_list('id', flat=True)

    bin = []
    for id in id_list:
        bin.append(id)
        if len(bin) >= batch_size:
            eb = ParaphraseBatch.objects.create(paraphrase_ids=json.dumps(bin))
            eb.save()
            bin.clear()

    eb = ParaphraseBatch.objects.create(paraphrase_ids=json.dumps(bin))
    eb.save()

if __name__ == '__main__':

    json_path = "data/relevant_senetences_to_perspectives/relevant_sentences_to_perspectives.json"

    update_paraphrase_table(json_path)
    generate_paraphrase_batches()

