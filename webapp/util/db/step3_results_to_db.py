import pandas as pd
from webapp.models import Step3Results

if __name__ == '__main__':

    results = "data/pilot14_step3_pilot/agreement_w_label.csv"
    df = pd.read_csv(results)

    for idx, row in df.iterrows():
        r = Step3Results.objects.create(perspective_id=row.perspective, evidence_id=row.evidence, vote_support=row.sup,
                                        vote_not_support=row.nsup, p_i=row.p_i, label=row.label)
        r.save()