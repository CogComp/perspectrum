from django.contrib.auth.models import User
from django.contrib.auth import login
from django.http import HttpResponse
from webapp.models import HITSession, PerspectiveRelation, Claim, EquivalenceHITSession
from django.db.models import Count
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_protect
from webapp.util.step2.equivalence_session_helpers import *

import json
import datetime
import random


def get_equivalence_hit_session(username):
    unfinished_sessions = EquivalenceHITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_equivalence_jobs(username, 10)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = EquivalenceHITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                                       instruction_complete=equivalence_instr_needed(username), duration=datetime.timedelta(),
                                                       last_start_time=time_now)

    return session

def equivalence_instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = EquivalenceHITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


# Number of assignments for each claim in the persp equivalence task
TARGET_ASSIGNMENT_COUNT = 3


def generate_equivalence_jobs(username, num_claims):
    """
    When each worker first login, generate the set of tasks they will be doing (for equivalence verification)
    :param username:
    :param num_claims:
    :return:
    """
    clean_equivalence_idle_sessions()

    claim_id_set = Claim.objects.filter(equivalence_assign_counts__lt=TARGET_ASSIGNMENT_COUNT)

    if len(claim_id_set) > num_claims:
        # Case 1: where we still have claims with lower than 3 assignments
        assign_counts = claim_id_set.values_list('id', 'equivalence_assign_counts', named=True)\
            .order_by('equivalence_assign_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in assign_counts:
            count_tuples.append((c.id, c.equivalence_assign_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_claims]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least 3 times.
        # Take 5 * num_claims least annotated claims
        assign_counts = Claim.objects.all().order_by('equivalence_assign_counts')\
            .values_list('id', 'equivalence_assign_counts', named=True)[:num_claims * 5]

        jobs = random.choices([t.id for t in assign_counts], k=10)

    if username != "TEST":
        increment_equivalence_assign_counts(jobs)

    return jobs