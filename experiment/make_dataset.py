"""
Converting raw db data into our desirable dataset format
(Not reading db data directly -- raw db data pre-extracted as csv)
"""
import pandas as pd
import json

# In File 1: claim db table output (csv)
claim_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/webapp_claim.csv"
# In File 2: evidence db table output (csv)
evidence_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/webapp_evidence.csv"
# In File 3: perspective db table output (csv)
persp_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/webapp_perspective.csv"
# In File 4: claim-perspective annotation db table output (csv)
persp_rel_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/webapp_perspectiverelation.csv"
# In File 5: perspective-evidence annotation db table output (csv)
evidence_rel_path = "/home/squirrel/ccg-new/projects/perspective/data/database_output/webapp_evidencerelation.csv"
# In File 6: IAA analysis result (csv) to filter out perspectives that are useless
persp_iaa_path = "/home/squirrel/ccg-new/projects/perspective/data/pilot1_persp_verification/persp_iaa.csv"

# Out File 1: Claims concatenated with corresponding perspectives, id = perspective id
claim_persp_out_path = "/home/squirrel/ccg-new/projects/perspective/data/pilot3_twowingos/110718-training-data/claim_perspective.json"
# Out File 2: Evidences w/ id
evidence_out_path = "/home/squirrel/ccg-new/projects/perspective/data/pilot3_twowingos/110718-training-data/evidence.json"
# Out File 3: Gold annotation between perspective and claim
gold_annotation_out_path = "/home/squirrel/ccg-new/projects/perspective/data/pilot3_twowingos/110718-training-data/gold_annotation.json"
# Out File 4: Perspective only version
persp_out_path = "/home/squirrel/ccg-new/projects/perspective/data/pilot3_twowingos/110718-training-data/perspective.json"

def concat_two_sentences(sent1, sent2):
    sent1 = sent1.strip()
    sent2 = sent2.strip()

    if sent1.endswith('.'):
        return sent1 + ' ' + sent2
    else:
        return sent1 + '. ' + sent2


if __name__ == '__main__':

    # Read them all in
    cdf = pd.read_csv(claim_path)
    edf = pd.read_csv(evidence_path)
    pdf = pd.read_csv(persp_path)
    p_rel_df = pd.read_csv(persp_rel_path)
    e_rel_df = pd.read_csv(evidence_rel_path)
    iaa_df = pd.read_csv(persp_iaa_path)

    p_rel_df = p_rel_df.loc[p_rel_df.author == 'GOLD']
    # Produce the evidence output, since it's the easiest
    edf[['id', 'content']].to_json(evidence_out_path, orient='records')

    # Produce the gold annotation
    valid_pdf = pdf.loc[pdf.pilot1_high_agreement == 1]
    valid_pids = set(valid_pdf.id.unique())

    p_c_merged_df = pd.merge(valid_pdf, p_rel_df[['claim_id', 'perspective_id', 'rel']], left_on='id', right_on='perspective_id')

    p_c_e_merged_df = pd.merge(p_c_merged_df, e_rel_df[['perspective_id', 'evidence_id']], on='perspective_id')
    p_c_e_merged_df.info()

    p_c_e_merged_df[['perspective_id', 'evidence_id']].to_json(gold_annotation_out_path, orient='records')

    # produce perspective + claim
    claim_persp_objects = []
    for idx, row in p_c_e_merged_df.iterrows():
        persp_title = row.title
        claim_title = cdf.loc[cdf.id == row.claim_id].iloc[0].title

        claim_persp_objects.append({
            'id': row.id,
            'title': concat_two_sentences(claim_title, persp_title)
        })

    with open(claim_persp_out_path, 'w') as fout:
        json.dump(claim_persp_objects, fout)

    persp_objects = []
    for idx, row in p_c_e_merged_df.iterrows():
        persp_title = row.title

        persp_objects.append({
            'id': row.id,
            'title': persp_title
        })

    with open(persp_out_path, 'w') as fout:
        json.dump(persp_objects, fout)