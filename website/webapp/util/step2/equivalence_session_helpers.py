from webapp.models import *
import datetime
import json
import collections

# Equivalence Verification Assignment counts

TIMEOUT_IDLE_MINUTE = 30

_TIMEOUT_IDLE_MINUTE = datetime.timedelta(minutes=TIMEOUT_IDLE_MINUTE)



def clean_equivalence_idle_sessions():
    sessions = EquivalenceHITSession.objects.filter(job_complete=0, finished_jobs="[]")
    for s in sessions.iterator():
        last_access_time = s.last_start_time
        time_now = datetime.datetime.now(datetime.timezone.utc)
        elapsed = time_now - last_access_time
        if elapsed > _TIMEOUT_IDLE_MINUTE:
            jobs = json.loads(s.jobs)
            decrease_equivalence_assign_counts(jobs)
            s.delete()

def increment_equivalence_assign_counts(claim_ids):
    """
    increase by assignment counts of the claim ids by 1
    :param claim_ids:
    :return:
    """
    return _offset_equivalence_assign_counts(claim_ids, 1)


def decrease_equivalence_assign_counts(claim_ids):
    """
    decrease by assignment counts of the claim ids by 1
    :param claim_ids:
    :return:
    """
    return _offset_equivalence_assign_counts(claim_ids, -1)


def _offset_equivalence_assign_counts(claim_ids, offset):
    for cid in claim_ids:
        claim = Claim.objects.get(id=cid)
        claim.evidence_assign_counts += offset
        claim.save()

