from django.contrib.auth.models import User
from django.contrib.auth import login
from django.http import HttpResponse
from webapp.models import HITSession, PerspectiveRelation, Claim, EvidenceHITSession
from django.db.models import Count
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_protect
from webapp.util.session_helpers import *

import json
import datetime
import hashlib
import random

@csrf_protect
def auth_login(request):
    """

    :param request
    :return:
    """
    username = request.POST['username']
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        user = User.objects.create_user(username)

    login(request=request, user=user)

    return HttpResponse(status=200) # The actual redirect happens front end, since ajax don't accept redirect response


def get_hit_session(username):
    unfinished_sessions = HITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_jobs(username, 10)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = HITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                            instruction_complete=instr_needed(username), duration=datetime.timedelta(),
                                            last_start_time=time_now)

    return session

def get_evidence_hit_session(username):
    unfinished_sessions = EvidenceHITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_evidence_jobs(username, 10)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = EvidenceHITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                            instruction_complete=evidence_instr_needed(username), duration=datetime.timedelta(),
                                            last_start_time=time_now)

    return session


def instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = HITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


def evidence_instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = EvidenceHITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


TARGET_ASSIGNMENT_PER_CLAIM = 3

def generate_jobs(username, num_claims):
    """
    :param username:
    :param num_claims: number of claims you want
    :return: list of claim ids
    """
    clean_idle_sessions()

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

    increment_assignment_counts(jobs)
    return jobs


# Number of assignments for each claim in the evidence verification task
ASSIGNMENT_PER_CLAIM_EVIDENCE = 3

def generate_evidence_jobs(username, num_claims):
    """
    When each worker first login, generate the set of tasks they will be doing (for evidence verification)
    :param username:
    :param num_claims:
    :return:
    """
    clean_evidence_idle_sessions()

    claim_id_set = Claim.objects.filter(evidence_assign_counts__lt=ASSIGNMENT_PER_CLAIM_EVIDENCE)

    if len(claim_id_set) > num_claims:
        # Case 1: where we still have claims with lower than 3 assignments
        assign_counts = claim_id_set.values_list('id', 'evidence_assign_counts', named=True)\
            .order_by('evidence_assign_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in assign_counts:
            count_tuples.append((c.id, c.evidence_assign_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_claims]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least 3 times.
        # Take 5 * num_claims least annotated claims
        assign_counts = Claim.objects.all().order_by('evidence_finished_counts')\
            .values_list('id', 'evidence_finished_counts', named=True)[:num_claims * 5]

        jobs = random.choices([t.id for t in assign_counts], k=10)

    increment_evidence_assign_counts(jobs)
    return jobs

