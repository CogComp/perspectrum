"""
Converting raw db data into our desirable dataset format
(Not reading db data directly -- raw db data pre-extracted as csv)
"""
import pandas as pd
import json

# In File 1: claim db table output (csv)
claim_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/re-step1/webapp_claim.csv"

# In File 2: perspective db table output (csv)
persp_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/re-step1/webapp_perspective.csv"

# In File 3: step 1 results
re_step1_result = "/home/squirrel/ccg-new/projects/perspective/data/database_output/re-step1/webapp_restep1results.csv"


# Out File 1: Claims concatenated with corresponding perspectives, id = perspective id
claim_out_path = "/home/squirrel/ccg-new/projects/perspective/data/dataset/claim.json"
# Out File 3: Gold annotation between perspective and claim
gold_annotation_out_path = "/home/squirrel/ccg-new/projects/perspective/data/dataset/gold_annotation.json"
# Out File 4: Perspective only version
persp_out_path = "/home/squirrel/ccg-new/projects/perspective/data/dataset/perspective.json"

def concat_two_sentences(sent1, sent2):
    sent1 = sent1.strip()
    sent2 = sent2.strip()

    if sent1.endswith('.'):
        return sent1 + ' ' + sent2
    else:
        return sent1 + '. ' + sent2


if __name__ == '__main__':

    # Read them all in
    cdf = pd.read_csv(claim_path).dropna()
    pdf = pd.read_csv(persp_path).dropna()
    re_df = pd.read_csv(re_step1_result)

    re_df = re_df[re_df.p_i_3 > 0.5].dropna()

    re_df.info()
    claims = re_df.claim_id.unique()
    perspectives = re_df.perspective_id.unique()

    # Output claims
    cdf = cdf[cdf["id"].isin(claims)]
    cdf[["id", "title"]].to_json(claim_out_path, orient='records')

    pdf = pdf[pdf["id"].isin(perspectives)]
    pdf[["id", "title"]].to_json(persp_out_path, orient='records')

    annos = []
    for _, row in re_df.iterrows():
        sup = row.vote_support + row.vote_leaning_support
        und = row.vote_leaning_undermine + row.vote_undermine
        ns = row.vote_not_valid

        votes = [sup, und, ns]
        stance = ["S", "U", "N"]
        i = votes.index(max(votes))
        annos.append({
            "claim_id" : int(row.claim_id),
            "perspective_id": int(row.perspective_id),
            "stance": stance[i]
        })

    with open(gold_annotation_out_path, 'w') as fout:
        json.dump(annos, fout)