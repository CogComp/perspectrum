from webapp.util.step2a.paraphrase_session_helpers import *

import json
import datetime
import random
import numpy as np


def get_paraphrase_hit_session(username):
    unfinished_sessions = ParaphraseHITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_paraphrase_jobs(username, 1)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = ParaphraseHITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                                      instruction_complete=paraphrase_instr_needed(username), duration=datetime.timedelta(),
                                                      last_start_time=time_now)

    return session


def paraphrase_instr_needed(username):
    """
    Check if a user need to take instruction
    :param username:
    :return: True if user need to take instruction
    """
    count = ParaphraseHITSession.objects.filter(username=username, instruction_complete=True).count()
    return count > 0


# Number of assignments for each claim in the persp paraphrase task
TARGET_ASSIGNMENT_COUNT = 1


def generate_paraphrase_jobs(username, num_paraphrases):
    """
    When each worker first login, generate the set of tasks they will be doing (for paraphrase verification)
    :param username:
    :param num_paraphrases:
    :return:
    """
    clean_paraphrase_idle_sessions()

    finished = get_all_finished_batches(username)

    eb_id_set = ParaphraseBatch.objects.filter(assign_counts__lt=TARGET_ASSIGNMENT_COUNT).exclude(id__in=finished)

    if len(eb_id_set) >= num_paraphrases:
        # Case 1: where we still have claims with lower than k assignments
        assign_counts = eb_id_set.values_list('id', 'assign_counts', named=True)\
            .order_by('assign_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in assign_counts:
            count_tuples.append((c.id, c.assign_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_paraphrases]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least k times.
        # Take 5 * num_claims least annotated claims
        assign_counts = ParaphraseBatch.objects.all().exclude(id__in=finished).order_by('finished_counts')\
            .values_list('id', 'finished_counts', named=True)[:num_paraphrases * 5]

        if len(assign_counts) == 0:
            jobs = []
        else:
            jobs = np.random.choice([t.id for t in assign_counts], size=num_paraphrases, replace=False).tolist()

    if username != "TEST":
        increment_paraphrase_assign_counts(jobs)

    return jobs
