"""
Cleaning idle sessions that hasn't been active for 30 minutes
Update the assignment counts in claim table of the db
"""

from django.db.models import Count
from webapp.models import *
import datetime
import json
import collections


TIMEOUT_IDLE_MINUTE = 30

_TIMEOUT_IDLE_MINUTE = datetime.timedelta(minutes=TIMEOUT_IDLE_MINUTE)

def clean_idle_sessions():
    sessions = HITSession.objects.filter(job_complete=0, finished_jobs="[]")
    for s in sessions.iterator():
        last_access_time = s.last_start_time
        time_now = datetime.datetime.now(datetime.timezone.utc)
        elapsed = time_now - last_access_time
        if elapsed > _TIMEOUT_IDLE_MINUTE:
            jobs = json.loads(s.jobs)
            decrease_assignment_counts(jobs)
            s.delete()

def update_all_claim_counts():
    """
    Update assignment counts of all claims according to annotations already in the database
    :return:
    """
    jobs_json = HITSession.objects.exclude(job_complete=0, finished_jobs="[]")\
        .values("jobs", "finished_jobs")

    jobs = []
    finished = []

    for c in jobs_json:
        jobs += json.loads(c["jobs"])
        finished += json.loads(c["finished_jobs"])

    counter = collections.Counter(jobs)
    finished_counter = collections.Counter(finished)
    for key, count in counter.items():
        claim = Claim.objects.get(id=key)
        claim.assignment_counts = count
        if key in finished_counter:
            claim.finished_counts = finished_counter[key]
        claim.save()


def increment_assignment_counts(claim_ids):
    """
    increase by assignment counts of the claim ids by 1
    :param claim_ids:
    :return:
    """
    return _offset_assignment_counts(claim_ids, 1)


def decrease_assignment_counts(claim_ids):
    """
    decrease by assignment counts of the claim ids by 1
    :param claim_ids:
    :return:
    """
    return _offset_assignment_counts(claim_ids, -1)


def _offset_assignment_counts(claim_ids, offset):
    for cid in claim_ids:
        claim = Claim.objects.get(id=cid)
        claim.assignment_counts += offset
        claim.save()


# Evidence Verification Assignment counts

def clean_evidence_idle_sessions():
    sessions = EvidenceHITSession.objects.filter(job_complete=0, finished_jobs="[]")
    for s in sessions.iterator():
        last_access_time = s.last_start_time
        time_now = datetime.datetime.now(datetime.timezone.utc)
        elapsed = time_now - last_access_time
        if elapsed > _TIMEOUT_IDLE_MINUTE:
            jobs = json.loads(s.jobs)
            decrease_evidence_assign_counts(jobs)
            s.delete()

def increment_evidence_assign_counts(claim_ids):
    """
    increase by assignment counts of the claim ids by 1
    :param claim_ids:
    :return:
    """
    return _offset_assignment_counts(claim_ids, 1)


def decrease_evidence_assign_counts(claim_ids):
    """
    decrease by assignment counts of the claim ids by 1
    :param claim_ids:
    :return:
    """
    return _offset_assignment_counts(claim_ids, -1)


def _offset_evidence_assign_counts(claim_ids, offset):
    for cid in claim_ids:
        claim = Claim.objects.get(id=cid)
        claim.evidence_assign_counts += offset
        claim.save()

if __name__ == '__main__':
    clean_idle_sessions()
    update_all_claim_counts()