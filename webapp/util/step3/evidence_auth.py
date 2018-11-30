from webapp.util.step3.evidence_session_helpers import *

import json
import datetime
import random
import numpy as np


def get_evidence_hit_session(username):
    unfinished_sessions = EvidenceHITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_evidence_jobs(username, 1)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = EvidenceHITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                                       instruction_complete=evidence_instr_needed(username), duration=datetime.timedelta(),
                                                       last_start_time=time_now)

    return session

def evidence_instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = EvidenceHITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


# Number of assignments for each claim in the persp evidence task
TARGET_ASSIGNMENT_COUNT = 4


def generate_evidence_jobs(username, num_evidences):
    """
    When each worker first login, generate the set of tasks they will be doing (for evidence verification)
    :param username:
    :param num_evidences:
    :return:
    """
    clean_evidence_idle_sessions()

    finished = get_all_finished_batches(username)

    eb_id_set = EvidenceBatch.objects.filter(assign_counts__lt=TARGET_ASSIGNMENT_COUNT).exclude(id__in=finished)

    if len(eb_id_set) >= num_evidences:
        # Case 1: where we still have claims with lower than 3 assignments
        assign_counts = eb_id_set.values_list('id', 'assign_counts', named=True)\
            .order_by('assign_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in assign_counts:
            count_tuples.append((c.id, c.assign_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_evidences]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least 3 times.
        # Take 5 * num_claims least annotated claims
        assign_counts = EvidenceBatch.objects.all().exclude(id__in=finished).order_by('finished_counts')\
            .values_list('id', 'finished_counts', named=True)[:num_evidences * 5]

        if len(assign_counts) == 0:
            jobs = []
        else:
            jobs = np.random.choice([t.id for t in assign_counts], size=num_evidences, replace=False).tolist()

    if username != "TEST":
        increment_evidence_assign_counts(jobs)

    return jobs

def generate_evidence_jobs_pilot(username, num_evidences):
    test_batch = [1624,1625,1626,1627,1628,1629,1630,1631,1632]

    finished = get_all_finished_batches(username)
    eb_id_set = EvidenceBatch.objects.filter(assign_counts__lt=5, id__in=test_batch).exclude(id__in=finished)
    print(eb_id_set)
    if len(eb_id_set) >= num_evidences:
        # Case 1: where we still have claims with lower than 3 assignments
        assign_counts = eb_id_set.values_list('id', 'assign_counts', named=True)\
            .order_by('assign_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in assign_counts:
            count_tuples.append((c.id, c.assign_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_evidences]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least 3 times.
        # Take 5 * num_claims least annotated claims
        assign_counts = EvidenceBatch.objects.filter(id__in=test_batch).exclude(id__in=finished)\
        .order_by('finished_counts').values_list('id', 'finished_counts', named=True)[:num_evidences * 5]

        if len(assign_counts) == 0:
            jobs = []
        else:
            jobs = np.random.choice([t.id for t in assign_counts], size=num_evidences, replace=False).tolist()

    if username != "TEST":
        increment_evidence_assign_counts(jobs)

    return jobs