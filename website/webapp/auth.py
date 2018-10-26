from django.contrib.auth.models import User
from django.contrib.auth import login
from django.http import HttpResponse
from webapp.models import HITSession, PerspectiveRelation, Claim
from django.db.models import Count
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_protect
from webapp.util.session_helpers import clean_idle_sessions, increment_assignment_counts, decrease_assignment_counts

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


def instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = HITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


# def generate_jobs(username, num_claims):
#     """
#     Temporary solution
#     :param username:
#     :param num_claims: number of claims you want
#     :return: list of claim ids
#     """
#     claim_id_set = Claim.objects.all().values_list('id', flat=True)
#
#     # Temporary solution for idebate TODO: delete this line
#     claim_id_set = [x for x in claim_id_set if x >= 155]
#
#     sessions = HITSession.objects.all().order_by("-id")
#
#     max_id = max(claim_id_set)
#     min_id = min(claim_id_set)
#     if sessions.count() == 0:
#         start = min_id
#     else:
#         prev_jobs = json.loads(sessions[0].jobs)
#         start = prev_jobs[-1] + 1
#
#     # Temporary solution for idebate TODO: delete this if statement
#     if start < min_id:
#         start = min_id
#
#     jobs = []
#     for i in range(num_claims):
#         jid = start + i
#         if jid > max_id:
#             jid = jid - max_id - 1 + min_id
#         jobs.append(jid)
#
#     return jobs

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


