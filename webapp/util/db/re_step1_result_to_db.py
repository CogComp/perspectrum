from webapp.models import *
import pandas as pd
import sys

LABEL_5 = ["S", "A", "B", "U", "N"]
LABEL_3 = ["S", "U", "N"]
DUPLICATE_VOTE_LABEL = "D"


def get_majority_vote_label(votes, label_set, duplicate_maj_vote_label=DUPLICATE_VOTE_LABEL):
    max_val = max(votes)
    if votes.count(max_val) > 1:
        return duplicate_maj_vote_label
    else:
        return label_set[votes.index(max_val)]


def update_labels():
    results = ReStep1Results.objects.all()

    for r in results:
        s = r.vote_support
        a = r.vote_leaning_support
        b = r.vote_leaning_undermine
        u = r.vote_undermine
        n = r.vote_not_valid
        votes_5 = [s, a, b, u, n]
        votes_3 = [s + a, b + u, n]
        r.label_3 = get_majority_vote_label(votes_3, LABEL_3)
        r.label_5 = get_majority_vote_label(votes_5, LABEL_5)
        r.save()

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python ... [origin_iaa_result] [google_iaa_result]", file=sys.stderr)
        exit(1)

    origin_iaa_result = sys.argv[1]
    google_iaa_result = sys.argv[2]

    google_df = pd.read_csv(google_iaa_result, engine="python")
    origin_df = pd.read_csv(origin_iaa_result, engine="python")

    origin_df.info()
    google_df.info()

    for idx, row in google_df.iterrows():
        r = ReStep1Results.objects.create(claim_id=row["claim"], perspective_id=row["perspective"], vote_support=row["sup"],
                                      vote_leaning_support=row["lsup"], vote_leaning_undermine=row["lund"],
                                      vote_undermine=row["und"], vote_not_valid=row["not_valid"], p_i_5=row["P_i_5_labels"],
                                      p_i_3=row["P_i_3_labels"])
        r.save()

    for idx, row in origin_df.iterrows():
        cid = PerspectiveRelation.objects.get(author=PerspectiveRelation.GOLD, perspective_id=row.perspective).claim_id
        r = ReStep1Results.objects.create(claim_id=cid, perspective_id=row["perspective"], vote_support=row["sup"],
                                      vote_leaning_support=row["lsup"], vote_leaning_undermine=row["lund"],
                                      vote_undermine=row["und"], vote_not_valid=row["not_valid"], p_i_5=row["P_i_5_labels"],
                                      p_i_3=row["P_i_3_labels"])
        r.save()

    update_labels()