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
            s.delete()

def update_assignment_counts():
    """
    Update assignment counts according to annotations already in the database
    :return:
    """
    jobs_json = HITSession.objects.exclude(job_complete=0, finished_jobs="[]")\
        .values("jobs")

    jobs = []

    for c in jobs_json:
        jobs += json.loads(c["jobs"])

    counter = collections.Counter(jobs)
    for key, count in counter.items():
        claim = Claim.objects.get(id=key)
        claim.assignment_counts = count
        claim.save()


if __name__ == '__main__':
    clean_idle_sessions()
    update_assignment_counts()