import datetime
import json
import random

from webapp.models import HITSession, Claim
from webapp.util.step1.session_helpers import clean_idle_persp_sessions, increment_persp_assignment_counts


def get_persp_hit_session(username):
    unfinished_sessions = HITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_persp_jobs(username, 10)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = HITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                            instruction_complete=persp_instr_needed(username), duration=datetime.timedelta(),
                                            last_start_time=time_now)

    return session


def persp_instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = HITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


TARGET_ASSIGNMENT_PER_CLAIM = 3

def generate_persp_jobs(username, num_claims):
    """
    :param username:
    :param num_claims: number of claims you want
    :return: list of claim ids
    """
    clean_idle_persp_sessions()

    claim_id_set = Claim.objects.filter(assignment_counts__lt=TARGET_ASSIGNMENT_PER_CLAIM)

    if len(claim_id_set) > num_claims:
        # Case 1: where we still have claims with lower than 3 assignments
        assign_counts = claim_id_set.values_list('id', 'assignment_counts', named=True)\
            .order_by('assignment_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in assign_counts:
            count_tuples.append((c.id, c.assignment_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_claims]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least 3 times.
        # Take 5 * num_claims least annotated claims
        assign_counts = Claim.objects.all().order_by('finished_counts')\
            .values_list('id', 'finished_counts', named=True)[:num_claims * 5]

        jobs = random.choices([t.id for t in assign_counts], k=10)

    increment_persp_assignment_counts(jobs)
    return jobs

