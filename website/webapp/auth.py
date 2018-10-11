from django.contrib.auth.models import User
from django.contrib.auth import login
from django.http import HttpResponse
from webapp.models import HITSession, PerspectiveRelation, Claim
from django.db.models import Count
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_protect

import json
import datetime
import hashlib

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


def generate_jobs(username, num_claims):
    """
    Get the list of ids of least annoatated claims in the database.
    Length of the list specified by num_claims.
    :param username:
    :param num_claims: number of claims you want
    :return: list of claim ids
    """
    PerspectiveRelation.objects.filter(author=username).values("claim_id")
    group = PerspectiveRelation.objects.exclude(author=PerspectiveRelation.GOLD)\
            .exclude(author="TEST").values("claim_id").annotate(count=Count("claim_id"))

    ranked = sorted(group, key=lambda entry: entry["count"])
    claim_id_list = [entry["claim_id"] for entry in ranked]

    ids_list = Claim.objects.all().values_list('id', flat=True)
    exclude_exist = [x for x in ids_list if x not in claim_id_list]
    ranked_all = exclude_exist + ranked

    return ranked_all[:num_claims]


def generate_code(username, time_now):
    """
    Generate the reward code for HIT session (8 bytes of hex)
    :param username:
    :param time_now: datetime object
    :return:
    """
    line = (username + str(time_now)).encode('utf-8')
    code = hashlib.md5(line).hexdigest()[:8]
    return code
