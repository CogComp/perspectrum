from webapp.util.step4.topic_session_helpers import *

import json
import datetime
import random
import numpy as np


def get_topic_hit_session(username):
    unfinished_sessions = TopicHITSession.objects.filter(username=username).exclude(job_complete=True)
    if unfinished_sessions.count() > 0:
        session = unfinished_sessions[0]
    else:
        claim_ids = generate_topic_jobs(username, 10)
        time_now = datetime.datetime.now(datetime.timezone.utc)
        session = TopicHITSession.objects.create(username=username, jobs=json.dumps(claim_ids), finished_jobs=json.dumps([]),
                                                       instruction_complete=1, duration=datetime.timedelta(),
                                                       last_start_time=time_now)

    return session



# Number of assignments for each claim in the persp topic task
TARGET_ASSIGNMENT_COUNT = 2


def generate_topic_jobs(username, num_claims):
    """
    When each worker first login, generate the set of tasks they will be doing (for topic verification)
    :param username:
    :param num_claims:
    :return:
    """
    clean_topic_idle_sessions()

    eb_id_set = Claim.objects.filter(topic_assign_counts__lt=TARGET_ASSIGNMENT_COUNT)

    if len(eb_id_set) >= num_claims:
        # Case 1: where we still have claims with lower than 3 assignments
        topic_assign_counts = eb_id_set.values_list('id', 'topic_assign_counts', named=True)\
            .order_by('topic_assign_counts')

        # Add [0, 1) random parts to each assignment counts, for randomly sorting claims with assignment counts
        count_tuples = []
        for c in topic_assign_counts:
            count_tuples.append((c.id, c.topic_assign_counts + random.random()))

        count_tuples = sorted(count_tuples, key=lambda t: t[1])[:num_claims]

        jobs = [t[0] for t in count_tuples]

    else:
        # Case 2: all claims are assigned at least 3 times.
        # Take 5 * num_claims least annotated claims
        topic_assign_counts = Claim.objects.all().order_by('finished_counts')\
            .values_list('id', 'finished_counts', named=True)[:num_claims * 5]

        if len(topic_assign_counts) == 0:
            jobs = []
        else:
            jobs = np.random.choice([t.id for t in topic_assign_counts], size=num_claims, replace=False).tolist()

    if username != "TEST":
        increment_topic_assign_counts(jobs)

    return jobs
