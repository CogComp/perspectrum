import json
import math
import zipfile
from io import BytesIO

from io import StringIO
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.core.mail import send_mail, BadHeaderError
from django.shortcuts import render, redirect
from django.core.files import File

from experiment.query_elasticsearch import get_perspective_from_pool
from experiment.query_elasticsearch import get_evidence_from_pool
from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, value, os

from experiment.run_bert_on_perspectrum import BertBaseline
from .forms import ContactForm

from webapp.models import *
from webapp.util.step1.persp_verification_auth import get_persp_hit_session
from webapp.util.step2b.equivalence_auth import get_equivalence_hit_session
from webapp.util.step2a.paraphrase_auth import get_paraphrase_hit_session
from webapp.util.step3.evidence_auth import get_evidence_hit_session
from webapp.util.step4.topic_auth import get_topic_hit_session

from collections import OrderedDict
import datetime
import random
from copy import deepcopy

import experiment.query_elasticsearch as es

file_names = {
    'evidence': 'data/dataset/evidence_pool_v0.2.json',
    'perspective': 'data/dataset/perspective_pool_v0.2.json',
    'claim_annotation': 'data/dataset/perspectrum_with_answers_v0.2.json'
}

"""Helper functions"""


def load_json(file_name):
    with open(file_name, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        return data


def save_json(data, file_name):
    with open(file_name, 'w') as data_file:
        _df = File(data_file)
        _df.write(json.dumps(data))


def get_pool_from_claim_id(claim_id):
    """
    TODO: Change this function! Right now it's only for testing purpose
    :param claim_id: id of the claim
    :return:
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id). \
                             order_by("?")[:2]
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


def get_all_persp(claim_id):
    """
    :param claim_id: id of the claim
    :return: list of perspectives
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id). \
        order_by("?")
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


def get_all_google_persp(claim_id):
    """
    :param claim_id: id of the claim
    :return: list of perspectives
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD,
                                                            claim_id=claim_id, comment="google") \
        .order_by("?")
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


def get_all_original_persp(claim_id):
    """
    :param claim_id: id of the claim
    :return: list of perspectives
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id) \
        .exclude(comment="google").order_by("?")
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


""" APIs """


def personality(request):
    context = {}
    return render(request, 'personality.html', context)


def main_page(request):
    context = {
        "datasets": list(file_names.keys())
    }
    return render(request, 'main.html', context)


def vis_claims(request):
    claim_titles = []
    claims = load_json(file_names["claim_annotation"])
    for c in claims:
        claim_titles.append((c["text"], c["cId"]))

    context = {
        "claim_titles": claim_titles
    }
    return render(request, 'claims.html', context)


def vis_spectrum(request, claim_id):
    claim = Claim.objects.get(id=claim_id)

    categories = [[] for _ in range(5)]

    pq = ReStep1Results.objects.filter(claim_id=claim.id, p_i_3__gte=0.5)
    for r in pq:
        p = Perspective.objects.get(id=r.perspective_id)
        votes = [r.vote_support, r.vote_leaning_support, r.vote_leaning_undermine, r.vote_undermine, r.vote_not_valid]
        idx = votes.index(max(votes))
        categories[idx].append(p)

    context = {
        "claim": claim,
        "sup": categories[0],
        "lsup": categories[1],
        "lund": categories[2],
        "und": categories[3],
        "nv": categories[4],
    }
    return render(request, 'step1/vis_spectrum.html', context)


# separated by commas
def vis_spectrum_js_list(request, claim_id_list):
    ids = claim_id_list.split('-')
    return vis_spectrum_js(request, [int(x) for x in ids])


def vis_spectrum_js_range(request, claim_id_range):
    split = claim_id_range.split('-')
    return vis_spectrum_js(request, list(range(int(split[0]), int(split[1]))))


def vis_spectrum_js_index(request, claim_id):
    # print(claim_id)
    # if claim_id != None and int(claim_id) >= 0:
    return vis_spectrum_js(request, [int(claim_id)])


def vis_spectrum_js(request, claim_ids_all):
    persps = load_json(file_names["perspective"])
    claims = load_json(file_names["claim_annotation"])

    persp_dict = {}
    claim_dict = {}
    for p in persps:
        persp_dict[p["pId"]] = p["text"]

    for c in claims:
        claim_dict[c["cId"]] = c

    # if claim_id != None and type(claim_id) == int and int(claim_id) >= 0:
    #     print("type 1: ")
    #     print(claim_id)
    #     print(int(claim_id))
    #     claim_ids_all = [int(claim_id)]
    # else:
    #     print("typ 2:")
    #     claim_ids_all = [142, 17] # list(claim_dict.keys())[20:3]

    print(claim_ids_all)

    used_evidences = []
    claim_persp_bundled = []
    for claim_id in claim_ids_all:
        c_title = claim_dict[claim_id]["text"]
        persp_sup = []
        persp_und = []
        for cluster_id, cluster in enumerate(claim_dict[claim_id]["perspectives"]):
            # titles = [str(pid) + ": " + persp_dict[pid] for pid in cluster["pids"]]
            evidences = cluster["evidence"]
            for pid in cluster["pids"]:
                title = str(pid) + ": " + persp_dict[pid]
                if cluster['stance_label_3'] == "SUPPORT":
                    persp_sup.append((title, pid, cluster_id + 1, evidences, 1.0))
                    used_evidences.extend(evidences)
                elif cluster['stance_label_3'] == "UNDERMINE":
                    persp_und.append((title, pid, cluster_id + 1, evidences, 1.0))
                    used_evidences.extend(evidences)
        claim_persp_bundled.append((c_title, persp_sup, persp_und))

    used_evidences_and_texts = [(e, evidence_dict[e]) for e in set(used_evidences)]

    # print(claim_persp_bundled)
    # print(used_evidences_and_texts)

    # context = {
    #     "claim": c_title,
    #     "persp_sup": persp_sup,
    #     "persp_und": persp_und,
    #     "used_evidences_and_texts": used_evidences_and_texts
    # }
    context = {
        "claim_persp_bundled": claim_persp_bundled,
        "used_evidences_and_texts": used_evidences_and_texts
    }

    return render(request, 'vis_dataset_js.html', context)


STANCE_FLIP_MAPPING = {
    "SUPPORT": "UNDERMINE",
    "MILDLY_SUPPORT": "MILDLY_UNDERMINE",
    "MILDLY_UNDERMINE": "MILDLY_SUPPORT",
    "UNDERMINE": "SUPPORT",
}


def dataset_download(request):
    prefix = "data/dataset/"
    filelist = [
        prefix + "dataset_split_v0.2.json",
        prefix + "evidence_pool_v0.2.json",
        prefix + "perspective_pool_v0.2.json",
        prefix + "perspectrum_with_answers_v0.2.json",
        prefix + "license.txt"
    ]

    byte_data = BytesIO()
    zip_file = zipfile.ZipFile(byte_data, "w")

    for file in filelist:
        filename = os.path.basename(os.path.normpath(file))
        zip_file.write(file, filename)
    zip_file.close()

    response = HttpResponse(byte_data.getvalue(), content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename=perspectrum_v0_2.zip'

    # Print list files in zip_file
    zip_file.printdir()

    return response


def dataset_page(request):
    context = {}
    return render(request, 'dataset_page.html', context)


## utils functions for the side-by-side view
def unify_persps(request, cid1, cid2, flip_stance):
    if cid1 == cid2:
        return HttpResponse("Success", status=200)

    claim1 = claim_dict[cid1]
    claim2 = claim_dict[cid2]

    for pp in claim1['perspectives']:
        pid = pp["pids"][0]
        add_perspective_to_claim(request, cid1, pid, cid2, flip_stance)

    for pp in claim2['perspectives']:
        pid = pp["pids"][0]
        add_perspective_to_claim(request, cid2, pid, cid1, flip_stance)

    return HttpResponse("Success", status=200)


def add_perspective_to_claim(request, cid_from, pid, cid_to, flip_stance):
    if cid_from == cid_to:
        return HttpResponse("Success", status=200)

    claim_from = claim_dict[cid_from]
    claim_to = claim_dict[cid_to]

    c1_idx = None
    for idx, p in enumerate(claim_from['perspectives']):
        if pid in p['pids']:
            c1_idx = idx

    c2_contains_pid = False
    for idx, p in enumerate(claim_to['perspectives']):
        if pid in p['pids']:
            c2_contains_pid = True
            break

    if (c1_idx != None) and not c2_contains_pid:
        cluster_cpy = deepcopy(claim_from['perspectives'][c1_idx])
        print(cluster_cpy)
        claim_to['perspectives'].append(cluster_cpy)
        if flip_stance:
            lbl3 = cluster_cpy['stance_label_3']
            if lbl3 in STANCE_FLIP_MAPPING:
                cluster_cpy['stance_label_3'] = STANCE_FLIP_MAPPING[lbl3]

            lbl5 = cluster_cpy['stance_label_5']
            if lbl5 in STANCE_FLIP_MAPPING:
                cluster_cpy['stance_label_5'] = STANCE_FLIP_MAPPING[lbl5]

                cluster_cpy['voter_counts'].reverse()

    return HttpResponse("Success", status=200)


def delete_cluster(request, claim_id, perspective_id):
    # in the claim file, drop the link to the perspective
    claim = claim_dict[claim_id]
    claim['perspectives'] = [p for p in claim['perspectives'] if perspective_id not in p["pids"]]

    return HttpResponse("Success", status=200)


def delete_perspective(request, claim_id, perspective_id):
    # in the claim file, drop the link to the perspective
    claim = claim_dict[claim_id]

    for persp in claim['perspectives']:
        if perspective_id in persp["pids"]:
            if len(persp["pids"]) == 1:
                delete_cluster(request, claim_id, perspective_id)
            else:
                del persp["pids"][persp["pids"].index(perspective_id)]

    return HttpResponse("Success", status=200)


def add_perspective_to_cluster(request, claim_id, cluster_id, persp_id_to_add):
    # in the claim file, drop the link to the
    claim = claim_dict[claim_id]

    delete_perspective(request, claim_id, persp_id_to_add)

    for persp in claim['perspectives']:
        if cluster_id in persp["pids"]:
            persp["pids"].append(persp_id_to_add)

    return HttpResponse("Success", status=200)


def split_perspective_from_cluster(request, claim_id, perspective_id):
    claim = claim_dict[claim_id]

    new_cluster = None
    for persp in claim['perspectives']:
        if perspective_id in persp["pids"]:
            new_cluster = deepcopy(persp)
            if len(persp["pids"]) == 1:
                delete_cluster(request, claim_id, perspective_id)
            else:
                del persp["pids"][persp["pids"].index(perspective_id)]
            new_cluster["pids"] = [perspective_id]

    if new_cluster:
        claim['perspectives'].append(new_cluster)

    return HttpResponse("Success", status=200)


def merge_perspectives(request, cid1, pid1, cid2, pid2):
    if (cid1 == cid2) and (pid1 == pid2):
        return HttpResponse("Success", status=200)

    claim1 = claim_dict[cid1]
    claim2 = claim_dict[cid2]

    c1_idx = None
    cluster_1 = None
    for idx, p in enumerate(claim1['perspectives']):
        if pid1 in p['pids']:
            cluster_1 = p
            c1_idx = idx

    c2_idx = None
    cluster_2 = None
    for idx, p in enumerate(claim2['perspectives']):
        if pid2 in p['pids']:
            cluster_2 = p
            c2_idx = idx

    print(cluster_1, cluster_2)
    if (c1_idx != None) and (c2_idx != None):
        print(pid1, pid2)

        merged_pid = list(set(cluster_1['pids'] + cluster_2['pids']))
        print(merged_pid)
        merged_evi_ids = list(set(cluster_1['evidence'] + cluster_2['evidence']))
        cluster_1['pids'] = deepcopy(merged_pid)
        cluster_2['pids'] = deepcopy(merged_pid)
        cluster_1['evidence'] = deepcopy(merged_evi_ids)
        cluster_2['evidence'] = deepcopy(merged_evi_ids)

    if cid1 == cid2:
        claim1['perspectives'].pop(claim1['perspectives'].index(cluster_1))

    return HttpResponse("Success", status=200)


def save_defualt_on_disk(request):
    return save_updated_claim_on_disk(request, "perspectrum_with_answers_v0.1.json")


def save_updated_claim_on_disk(request, file_name):
    # convert map to list before saving
    claims_local = [claim_dict[k] for k in claim_dict.keys()]
    save_json(claims_local, "data/dataset/" + file_name)
    return HttpResponse("Success", status=200)


persps = load_json(file_names["perspective"])
claims = load_json(file_names["claim_annotation"])
evidence = load_json(file_names["evidence"])

persp_dict = {}
claim_dict = {}
evidence_dict = {}
for p in persps:
    persp_dict[p["pId"]] = p["text"]

for e in evidence:
    evidence_dict[e["eId"]] = e["text"]

for c in claims:
    claim_dict[c["cId"]] = c


def vis_dataset_side_by_side(request, claim_id1, claim_id2):
    # claim_id1 = 300
    claim_id1 = int(claim_id1)

    c_title1 = claim_dict[claim_id1]["text"]
    persp_sup1 = []
    persp_und1 = []
    for cluster in claim_dict[claim_id1]["perspectives"]:
        titles = [(pid, persp_dict[pid]) for pid in cluster["pids"]]

        if cluster['stance_label_3'] == "SUPPORT":
            persp_sup1.append(titles)
        elif cluster['stance_label_3'] == "UNDERMINE":
            persp_und1.append(titles)

    claim_id2 = int(claim_id2)

    c_title2 = claim_dict[claim_id2]["text"]
    persp_sup2 = []
    persp_und2 = []
    for cluster in claim_dict[claim_id2]["perspectives"]:
        titles = [(pid, persp_dict[pid]) for pid in cluster["pids"]]

        if cluster['stance_label_3'] == "SUPPORT":
            persp_sup2.append(titles)
        elif cluster['stance_label_3'] == "UNDERMINE":
            persp_und2.append(titles)

    context = {
        "cid1": claim_id1,
        "claim1": c_title1,
        "persp_sup1": persp_sup1,
        "persp_und1": persp_und1,
        "cid2": claim_id2,
        "claim2": c_title2,
        "persp_sup2": persp_sup2,
        "persp_und2": persp_und2,
    }

    return render(request, 'vis_dataset_side_by_side.html', context)


def vis_dataset(request, claim_id):
    persps = load_json(file_names["perspective"])
    claims = load_json(file_names["claim_annotation"])

    claim_id = int(claim_id)
    persp_dict = {}
    claim_dict = {}
    for p in persps:
        persp_dict[p["pId"]] = p["text"]

    for c in claims:
        claim_dict[c["cId"]] = c

    c_title = claim_dict[claim_id]["text"]
    persp_sup = []
    persp_und = []
    for cluster in claim_dict[claim_id]["perspectives"]:
        titles = [str(pid) + ": " + persp_dict[pid] for pid in cluster["pids"]]

        if cluster['stance_label_3'] == "SUPPORT":
            persp_sup.append(titles)
        elif cluster['stance_label_3'] == "UNDERMINE":
            persp_und.append(titles)

    context = {
        "claim": c_title,
        "persp_sup": persp_sup,
        "persp_und": persp_und,
    }

    return render(request, 'vis_dataset.html', context)


def vis_persps(request, claim_id):
    c_title = claim_dict[claim_id]["text"]
    persp_sup = []
    persp_und = []
    for cluster in claim_dict[claim_id]["perspectives"]:
        titles = [(pid, persp_dict[pid]) for pid in cluster["pids"]]

        if cluster['stance_label_3'] == "SUPPORT":
            persp_sup.append((titles, cluster["voter_counts"], cluster["evidence"]))
        elif cluster['stance_label_3'] == "UNDERMINE":
            persp_und.append((titles, cluster["voter_counts"], cluster["evidence"]))

    context = {
        "claim": c_title,
        "claim_id": claim_id,
        "persp_sup": persp_sup,
        "persp_und": persp_und,
    }

    return render(request, 'persp.html', context)


def vis_evidence(request, evidence_ids):
    eid_list = [int(eid) for eid in evidence_ids.split("-") if eid]
    _evidences = []

    for eid in eid_list:
        if eid in evidence_dict:
            _evidences.append((eid, evidence_dict[eid]))

    context = {
        "evidences": _evidences
    }
    return render(request, 'evidence.html', context)


@login_required
def vis_neg_anno(request, claim_id):
    return render(request, 'step1/claim_neg_anno.html', {})


@login_required
def vis_relation(request, claim_id):
    try:
        claim = Claim.objects.get(id=claim_id)
    except Claim.DoesNotExist:
        pass  # TODO: Do something? 404?

    perspective_pool = get_pool_from_claim_id(claim_id)

    return render(request, 'step1/claim_relation.html', {
        "claim": "",
        "perspective_pool": perspective_pool
    })


@login_required
@csrf_protect
def submit_instr(request):
    if request.method != 'POST':
        raise ValueError("submit_instr API only supports POST request")
        # TODO: Actaully not sure what to do here..
    else:
        username = request.user.username
        session = get_persp_hit_session(username)

        session.instruction_complete = True
        session.save()
        return HttpResponse("Submission Success!", status=200)


@login_required
@csrf_protect
def submit_rel_anno(request):
    """
    Accepts POST requests and update the annotations
    """
    if request.method != 'POST':
        raise ValueError("submit_rel_anno API only supports POST request")
        # TODO: Actaully not sure what to do here..
    else:
        claim_id = request.POST.get('claim_id')
        annos = request.POST.getlist('annotations[]')

        username = request.user.username
        session = get_persp_hit_session(username)

        if claim_id and annos:
            for a in annos:
                parts = a.split(',')
                if len(parts) != 2:
                    return HttpResponse("Submission Failed! Annotation not valid.", status=400)

                persp_id = parts[0]
                rel = parts[1]
                anno_entry = PerspectiveRelation.objects.create(author=username, claim_id=claim_id,
                                                                perspective_id=persp_id, rel=rel, comment="turk_google")
                anno_entry.save()

        # Update finished jobs in user session
        fj = set(json.loads(session.finished_jobs))
        fj.add(int(claim_id))
        session.finished_jobs = json.dumps(list(fj))

        # increment duration in database
        delta = datetime.datetime.now(datetime.timezone.utc) - session.last_start_time
        session.duration = session.duration + delta
        session.save()

        # Increment finished counts in claim table
        if username != 'TEST':
            claim = Claim.objects.get(id=claim_id)
            claim.finished_counts += 1
            claim.save()

        return HttpResponse("Submission Success!", status=200)


def render_login_page(request):
    """
    Renderer for login page
    """
    return render(request, "login.html", {})


from django.contrib.auth import logout


def logout_request(request):
    logout(request)
    return render_login_page(request)


@login_required
def render_list_page(request):
    """
    Renderer the list of task
    """
    username = request.user.username
    session = get_persp_hit_session(username)
    instr_complete = session.instruction_complete
    jobs = json.loads(session.jobs)
    finished = json.loads(session.finished_jobs)

    task_list = []
    for job in jobs:
        task_list.append({
            "id": job,
            "done": job in finished
        })

    tasks_are_done = all(item["done"] for item in task_list)

    task_id = -1
    if tasks_are_done:  # TODO: change this condition to if the user has completed the task
        task_id = session.id
        session.job_complete = True
        session.save()

    context = {"task_id": task_id, "instr_complete": instr_complete, "task_list": task_list}
    return render(request, "step1/list_tasks.html", context)


def render_instructions(request):
    return render(request, "step1/instructions.html", {})


def render_contact(request):
    if request.method == 'GET':
        form = ContactForm()
    else:
        form = ContactForm(request.POST)
        if form.is_valid():
            subject = form.cleaned_data['subject']
            from_email = form.cleaned_data['from_email']
            message = form.cleaned_data['message']
            try:
                send_mail(subject, message, from_email, ['sihaoc@seas.upenn.edu', 'danielkh@cis.upenn.edu'])
            except BadHeaderError:
                return HttpResponse('Invalid header found.')
            return redirect('success')
    return render(request, "contact.html", {'form': form})


def successView(request):
    return HttpResponse('Success! Thank you for your message.')


@login_required
def vis_normalize_persp(request, claim_id):
    username = request.user.username
    session = get_persp_hit_session(username)
    session.last_start_time = datetime.datetime.now(datetime.timezone.utc)
    session.save()
    try:
        claim = Claim.objects.get(id=claim_id)
    except Claim.DoesNotExist:
        pass  # TODO: Do something? 404?

    perspective_pool = get_all_persp(claim_id)

    return render(request, 'step1/normalize_persp.html', {
        "claim": claim,
        "perspective_pool": perspective_pool
    })


###############################################
#     STEP 2a APIs
#     Perspective Paraphrase
###############################################
@login_required
def render_step2a_instructions(request):
    return render(request, "step2a/instructions.html", {})


@login_required
def render_step2a_task_list(request):
    username = request.user.username
    session = get_paraphrase_hit_session(username)

    instr_complete = session.instruction_complete
    jobs = json.loads(session.jobs)
    finished = json.loads(session.finished_jobs)

    task_list = []
    for job in jobs:
        task_list.append({
            "id": job,
            "done": job in finished
        })

    tasks_are_done = all(item["done"] for item in task_list)

    task_id = -1
    if tasks_are_done:  # TODO: change this condition to if the user has completed the task
        task_id = session.id
        session.job_complete = True
        session.save()

    context = {"task_id": task_id, "instr_complete": instr_complete, "task_list": task_list}

    return render(request, 'step2a/task_list.html', context)


@login_required
def vis_perspective_paraphrase(request, batch_id):
    username = request.user.username
    session = get_paraphrase_hit_session(username)
    session.last_start_time = datetime.datetime.now(datetime.timezone.utc)
    session.save()
    try:
        pb = ParaphraseBatch.objects.get(id=batch_id)
    except ParaphraseBatch.DoesNotExist:
        return HttpResponse(content="Batch_Id = {} Not found in the database!".format(batch_id), status=404)

    ppids = json.loads(pb.paraphrase_ids)

    paraphrases = [PerspectiveParaphrase.objects.get(id=i) for i in ppids]
    perspectives = []
    hints = {}
    claims = {}

    for pp in paraphrases:
        pid = pp.perspective_id
        perspectives.append(Perspective.objects.get(id=pid))
        cid = PerspectiveRelation.objects.filter(author="GOLD").exclude(comment="google").get(
            perspective_id=pid).claim_id
        c = Claim.objects.get(id=cid)

        _h = json.loads(pp.hints)
        random.shuffle(_h)
        hints[pid] = _h
        claims[pid] = c.title

    return render(request, 'step2a/paraphrase_perspectives.html', {
        "perspectives": perspectives,
        "hints": hints,
        "claims": claims
    })


@login_required
@csrf_protect
def step2a_submit_instr(request):
    if request.method != 'POST':
        raise ValueError("submit_instr API only supports POST request")
    else:
        username = request.user.username
        session = get_paraphrase_hit_session(username)

        session.instruction_complete = True
        session.save()
        return HttpResponse("Submission Success!", status=200)


@login_required
@csrf_protect
def submit_paraphrase_annotation(request):
    if request.method != 'POST':
        raise ValueError("submit_rel_anno API only supports POST request")
    else:
        batch_id = request.POST.get('batch_id')
        annos = json.loads(request.POST.get('annotations'))
        username = request.user.username
        session = get_paraphrase_hit_session(username)

        # Update annotation in EquivalenceAnnotation table
        for pid, paras in annos.items():
            p = PerspectiveParaphrase.objects.get(perspective_id=pid)
            user_para_str = p.user_generated.replace('"', '\\"').replace('[\'', '["').replace('\']', '"]').replace(
                '\', \'', '", "') \
                .replace('\', "', '", "').replace('", \'', '", "')
            print(user_para_str)
            user_para = json.loads(user_para_str)
            sid_str = p.session_ids.replace('[\'', '["').replace('\']', '"]').replace('\', \'', '", "')
            session_ids = json.loads(sid_str)

            for pp in paras:
                user_para.append(pp)
                session_ids.append(session.id)

            p.user_generated = json.dumps(user_para)
            p.session_ids = json.dumps(session_ids)
            p.save()

        # Update finished jobs in user session
        fj = set(json.loads(session.finished_jobs))
        fj.add(int(batch_id))
        session.finished_jobs = json.dumps(list(fj))

        # increment duration in database
        delta = datetime.datetime.now(datetime.timezone.utc) - session.last_start_time
        session.duration = session.duration + delta
        session.save()

        # Increment finished assignment count in claim table, if not using test acc
        if username != 'TEST':
            c = ParaphraseBatch.objects.get(id=batch_id)
            c.finished_counts += 1
            c.save()

        return HttpResponse("Submission Success!", status=200)


###############################################
#     STEP 2b APIs
#     Perspective Equivalence
###############################################
@login_required
def render_step2b_instructions(request):
    return render(request, "step2b/instructions.html", {})


@login_required
def render_step2b_task_list(request):
    username = request.user.username
    session = get_equivalence_hit_session(username)

    instr_complete = session.instruction_complete
    jobs = json.loads(session.jobs)
    finished = json.loads(session.finished_jobs)

    task_list = []
    for job in jobs:
        task_list.append({
            "id": job,
            "done": job in finished
        })

    tasks_are_done = all(item["done"] for item in task_list)

    task_id = -1
    if tasks_are_done:  # TODO: change this condition to if the user has completed the task
        task_id = session.id
        session.job_complete = True
        session.save()

    context = {"task_id": task_id, "instr_complete": instr_complete, "task_list": task_list}

    return render(request, 'step2b/task_list.html', context)


@login_required
def vis_persp_equivalence(request, batch_id):
    username = request.user.username
    session = get_equivalence_hit_session(username)
    session.last_start_time = datetime.datetime.now(datetime.timezone.utc)
    session.save()

    try:
        eb = EquivalenceBatch.objects.get(id=batch_id)
    except EquivalenceBatch.DoesNotExist:
        return HttpResponse(content="Batch_Id = {} Not found in the database!".format(batch_id), status=404)

    persp_ids = json.loads(eb.perspective_ids)

    persps = []
    claims = {}
    candidates = {}

    for cid, pid in persp_ids:
        persp = Perspective.objects.get(id=pid)
        persps.append(persp)

        claims[pid] = Claim.objects.get(id=cid).title

        cand_ids = set(json.loads(Perspective.objects.get(id=pid).similar_persps))
        cand_persps = Perspective.objects.filter(id__in=cand_ids)
        candidates[pid] = cand_persps

    return render(request, 'step2b/persp_equivalence.html', {
        "claims": claims,
        "perspective_pool": persps,
        "candidates": candidates
    })


@login_required
@csrf_protect
def submit_equivalence_annotation(request):
    if request.method != 'POST':
        raise ValueError("submit_rel_anno API only supports POST request")
    else:
        claim_id = request.POST.get('claim_id')
        annos = json.loads(request.POST.get('annotations'))
        username = request.user.username
        session = get_equivalence_hit_session(username)

        # Update annotation in EquivalenceAnnotation table
        for p, cands in annos.items():
            EquivalenceAnnotation.objects.create(session_id=session.id, author=username,
                                                 perspective_id=p, user_choice=json.dumps(annos[p]))

        # Update finished jobs in user session
        fj = set(json.loads(session.finished_jobs))
        fj.add(int(claim_id))
        session.finished_jobs = json.dumps(list(fj))

        # increment duration in database
        delta = datetime.datetime.now(datetime.timezone.utc) - session.last_start_time
        session.duration = session.duration + delta
        session.save()

        # Increment finished assignment count in claim table, if not using test acc
        if username != 'TEST':
            eb = EquivalenceBatch.objects.get(id=claim_id)
            eb.finished_counts += 1
            eb.save()

        return HttpResponse("Submission Success!", status=200)


@login_required
@csrf_protect
def step2b_submit_instr(request):
    if request.method != 'POST':
        raise ValueError("submit_instr API only supports POST request")
    else:
        username = request.user.username
        session = get_equivalence_hit_session(username)

        session.instruction_complete = True
        session.save()
        return HttpResponse("Submission Success!", status=200)


###############################################
#     STEP 3 APIs
#     Evidence verification
###############################################

PERSP_NUM = 8


@login_required
def render_evidence_verification(request, batch_id):
    username = request.user.username
    session = get_evidence_hit_session(username)
    session.last_start_time = datetime.datetime.now(datetime.timezone.utc)
    session.save()
    try:
        eb = EvidenceBatch.objects.get(id=batch_id)
    except EvidenceBatch.DoesNotExist:
        pass  # TODO: Do something? 404?

    eids = json.loads(eb.evidence_ids)

    valid_persp_ids = ReStep1Results.objects.filter(label_3__in=["S", "U"], p_i_3__gt=0.5).values_list(
        'perspective_id').distinct()

    evidences = [Evidence.objects.get(id=i) for i in eids]
    keywords = {}
    candidates = {}
    for evi in evidences:
        origin_cands = json.loads(evi.origin_candidates)
        google_cands = json.loads(evi.google_candidates)

        same_claim_cands = []

        # Get Keywords
        try:
            _er = EvidenceRelation.objects.filter(author="GOLD").get(evidence_id=evi.id)
            pid = _er.perspective_id
            _pr = PerspectiveRelation.objects.filter(author="GOLD").get(perspective_id=pid)
            cid = _pr.claim_id
            _c = Claim.objects.get(id=cid)
            same_claim_cands = list(PerspectiveRelation.objects.filter(author="GOLD", claim_id=cid).
                                    exclude(comment="google").values_list("perspective_id", flat=True))

            same_claim_cands = [scc for scc in same_claim_cands if scc in valid_persp_ids]

            _keywords = json.loads(_c.keywords)
        except EvidenceRelation.DoesNotExist:
            _keywords = []
        except PerspectiveRelation.DoesNotExist:
            _keywords = []
        except Claim.DoesNotExist:
            _keywords = []

        keywords[evi.id] = _keywords

        all_cands = origin_cands + same_claim_cands + google_cands
        cands = list(OrderedDict.fromkeys(all_cands))[:PERSP_NUM]

        persps = [Perspective.objects.get(id=i) for i in cands]

        # shuffle the order of perspectives
        candidates[evi.id] = persps

    return render(request, 'step3/evidence_verification.html', {
        "evidences": evidences,
        "candidates": candidates,
        "keywords": keywords
    })


def render_step3_task_list(request):
    username = request.user.username
    session = get_evidence_hit_session(username)

    instr_complete = session.instruction_complete
    jobs = json.loads(session.jobs)
    finished = json.loads(session.finished_jobs)

    task_list = []
    for job in jobs:
        task_list.append({
            "id": job,
            "done": job in finished
        })

    tasks_are_done = all(item["done"] for item in task_list)

    task_id = -1
    if tasks_are_done and len(jobs) > 0:  # TODO: change this condition to if the user has completed the task
        task_id = session.id
        session.job_complete = True
        session.save()

    context = {"task_id": task_id, "instr_complete": instr_complete, "task_list": task_list}

    return render(request, 'step3/task_list.html', context)


def render_step3_instructions(request):
    return render(request, "step3/instructions.html", {})


evidence_label_mapping = {
    "sup": "S",
    "nsup": "N"
}


@login_required
@csrf_protect
def submit_evidence_annotation(request):
    if request.method != 'POST':
        raise ValueError("submit_rel_anno API only supports POST request")
    else:
        batch_id = request.POST.get('batch_id')
        annos = json.loads(request.POST.get('annotations'))
        username = request.user.username
        session = get_evidence_hit_session(username)

        # Update annotation in EquivalenceAnnotation table
        for anno in annos:
            label = anno[2]
            if label in evidence_label_mapping:
                EvidenceRelation.objects.create(author=username, evidence_id=anno[0], perspective_id=anno[1],
                                                anno=evidence_label_mapping[label], comment="step3")

        # Update finished jobs in user session
        fj = set(json.loads(session.finished_jobs))
        fj.add(int(batch_id))
        session.finished_jobs = json.dumps(list(fj))

        # increment duration in database
        delta = datetime.datetime.now(datetime.timezone.utc) - session.last_start_time
        session.duration = session.duration + delta
        session.save()

        # Increment finished assignment count in claim table, if not using test acc
        if username != 'TEST':
            c = EvidenceBatch.objects.get(id=batch_id)
            c.finished_counts += 1
            c.save()

        return HttpResponse("Submission Success!", status=200)


@login_required
@csrf_protect
def step3_submit_instr(request):
    if request.method != 'POST':
        raise ValueError("submit_instr API only supports POST request")
    else:
        username = request.user.username
        session = get_evidence_hit_session(username)

        session.instruction_complete = True
        session.save()
        return HttpResponse("Submission Success!", status=200)


@login_required
def bert_baseline(request, claim_text=""):
    print(claim_text)
    if claim_text != "":
        claim = claim_text  #

        prob = LpProblem("perspectiveOptimization", LpMaximize)

        # given a claim, extract perspectives
        perspective_given_claim = [(p_text, pId, pScore / len(p_text.split(" "))) for p_text, pId, pScore in
                                   get_perspective_from_pool(claim, 30)]

        # create binary variables per perspective
        perspective_variables = []
        perspective_weights = []
        perspective_ids = []
        perspective_given_claim_subset = []

        for (pIdx, (p_text, pId, pScore)) in enumerate(perspective_given_claim):
            if pScore > 1.55:
                x = LpVariable("p" + str(pIdx), 0, 1)
                perspective_variables.append(x)
                perspective_weights.append(pScore)
                perspective_ids.append(pId)
                perspective_given_claim_subset.append((p_text, pId, pScore))
                # print(pScore)

        # print(len(perspective_variables))

        total_obj = sum(x * obj for x, obj in zip(perspective_variables, perspective_weights))

        # maximum and minimum number of perspectives selected
        total_weight = sum(x * 1.0 for x in perspective_variables)
        prob += total_weight <= 10
        # prob += total_weight >= 5

        assert (len(perspective_variables) == len(perspective_weights))

        # given a perspective, retrieve relevant perspectives
        # create a map of clustering similarity
        # offset = 7
        # pp_threshold = 10
        # lucene_perspective_pair_cache = {}
        # for p_text1, pId1, _ in perspective_given_claim:
        #     lucene_perspectives = get_perspective_from_pool(p_text1, 30)
        #     for p_text2, pId2, pScore2 in lucene_perspectives:
        #         # print(f"text1: {p_text1} - text2: {p_text2}:  {pScore2}")
        #         lucene_perspective_pair_cache[(pId1, pId2)] = pScore2 # / (1.0 * len(p_text2.split(" ")))

        # print(lucene_perspective_pair_cache)

        # perspective_pair_variables = []
        # perspective_pair_weights = []
        # # perspective_pair_variables_map = {}
        # for i1, (_, pId1, _) in enumerate(perspective_given_claim):
        #     # perspective_pair_variables_map[pId1] = []
        #     for i2, (_, pId2, _) in enumerate(perspective_given_claim):
        #         if pId1 == pId2 or (pId1, pId2) not in lucene_perspective_pair_cache:
        #             continue
        #         y = LpVariable("p" + str(pId1) + "-" + str(pId2), 0, 1)
        #         if (pId1, pId2) in lucene_perspective_pair_cache:
        #             # print("negative . . . ")
        #             score = lucene_perspective_pair_cache[(pId1, pId2)]
        #             if score > pp_threshold:
        #                 perspective_pair_variables.append(y)
        #                 perspective_pair_weights.append(7- 0.0001 * score)
        #                 # perspective_pair_variables_map[(pId1, pId2)] = score
        #                 prob += perspective_variables[i1] >= y
        #                 prob += perspective_variables[i2] >= y
        #                 prob += y >= (perspective_variables[i2] + perspective_variables[i1] - 1)
        #
        # total_obj += sum(x * obj for x, obj in zip(perspective_pair_variables, perspective_pair_weights))

        # print(perspective_pair_variables)

        # assert len(perspective_pair_variables) == len(perspective_pair_weights)

        # given perspectives, retrieve relevant evidences
        # evidence_variables = []
        # evidence_weights = []
        # evidence_ids = []
        # threshold = 10
        # perspective_to_evidence_variables = {}
        evidence_to_perspective_variables = {}
        # for p_text1, pId1, _ in perspective_given_claim:
        # perspective_to_evidence_variables[pId1] = []
        # lucene_evidences = get_evidence_from_pool(p_text1, 1)
        # for (eIdx, (e_text, eId, eScore)) in enumerate(lucene_evidences):
        # if pScore > threshold:
        #     x = LpVariable("pe" + str(pId1) + "-" + str(eId), 0, 1)
        # evidence_variables.append(x)
        # evidence_weights.append(pScore)
        # evidence_ids.append(eId)
        # perspective_to_evidence_variables[pId1].append((x, pScore, eId))
        # print(pScore)

        # if a perspective is active, it should be connected to at least one evidence
        # evidence is not active, unless it is connected to sth

        prob += total_obj
        status = prob.solve()
        print(LpStatus[status])

        # extract active variables
        used_evidences_and_texts = []
        persp_sup = []
        for pVar, p in zip(perspective_variables, perspective_given_claim_subset):
            pScore = p[2]
            p_text = p[0]
            # print(pVar)
            # print(value(pVar))
            if value(pVar) != None:
                if value(pVar) > 0.5:
                    lucene_evidences = get_evidence_from_pool(claim + p_text, 2)
                    evidences = []
                    if len(lucene_evidences) > 0:
                        (e_text, eId, eScore) = lucene_evidences[0]
                        evidences = eId
                        used_evidences_and_texts.append([eId, e_text.replace("`", "'")])
                    persp_sup.append((p[0], p[1], 1, [evidences], pScore))

            else:
                print("value is none")

        claim_persp_bundled = [(claim, persp_sup, [])]

        context = {
            "claim_persp_bundled": claim_persp_bundled,
            "used_evidences_and_texts": used_evidences_and_texts
        }
    else:
        context = {}

    return render(request, "vis_dataset_js.html", context)


### loading the BERT solvers
bb = BertBaseline(task_name="perspectrum_relevane", saved_model="/Users/daniel/ideaProjects/perspective/model/relevance/perspectrum_relevance_lr2e-05_bs32_epoch-0.pth", no_cuda=True)

@login_required
def perspectrum_solver(request, claim_text="", vis_type=""):
    """
    solves a given instances with one of the baselines.
    :param request: the default request argument.
    :param claim_text: the text of the input claim.
    :param vis_type: whether we visualize with the fancy graphical interface or we use a simple visualization.
    :param baseline_name: the solver name (BERT and Lucene).
    :return:
    """
    print(claim_text)
    if claim_text != "":
        claim = claim_text  #

        prob = LpProblem("perspectiveOptimization", LpMaximize)

        # given a claim, extract perspectives
        perspective_given_claim = [(p_text, pId, pScore / len(p_text.split(" "))) for p_text, pId, pScore in
                                   get_perspective_from_pool(claim, 30)]

        perspectives_sorted = [(p_text, pId, normalize(luceneScore), normalize(bb.predict(claim, p_text)[0])) for (p_text, pId, luceneScore) in
                               perspective_given_claim]

        perspectives_sorted = sorted(perspectives_sorted, key=lambda x: -x[3])

        # create binary variables per perspective
        perspective_variables = []
        perspective_weights = []
        perspective_ids = []
        perspective_given_claim_subset = []

        for (pIdx, (p_text, pId, pScore)) in enumerate(perspective_given_claim):
            if pScore > 1.55:
                x = LpVariable("p" + str(pIdx), 0, 1)
                perspective_variables.append(x)
                perspective_weights.append(pScore)
                perspective_ids.append(pId)
                perspective_given_claim_subset.append((p_text, pId, pScore))
                # print(pScore)

        total_obj = sum(x * obj for x, obj in zip(perspective_variables, perspective_weights))

        # maximum and minimum number of perspectives selected
        total_weight = sum(x * 1.0 for x in perspective_variables)
        prob += total_weight <= 10
        # prob += total_weight >= 5

        assert (len(perspective_variables) == len(perspective_weights))

        # given a perspective, retrieve relevant perspectives
        # create a map of clustering similarity
        # offset = 7
        # pp_threshold = 10
        # lucene_perspective_pair_cache = {}
        # for p_text1, pId1, _ in perspective_given_claim:
        #     lucene_perspectives = get_perspective_from_pool(p_text1, 30)
        #     for p_text2, pId2, pScore2 in lucene_perspectives:
        #         # print(f"text1: {p_text1} - text2: {p_text2}:  {pScore2}")
        #         lucene_perspective_pair_cache[(pId1, pId2)] = pScore2 # / (1.0 * len(p_text2.split(" ")))

        # print(lucene_perspective_pair_cache)

        # perspective_pair_variables = []
        # perspective_pair_weights = []
        # # perspective_pair_variables_map = {}
        # for i1, (_, pId1, _) in enumerate(perspective_given_claim):
        #     # perspective_pair_variables_map[pId1] = []
        #     for i2, (_, pId2, _) in enumerate(perspective_given_claim):
        #         if pId1 == pId2 or (pId1, pId2) not in lucene_perspective_pair_cache:
        #             continue
        #         y = LpVariable("p" + str(pId1) + "-" + str(pId2), 0, 1)
        #         if (pId1, pId2) in lucene_perspective_pair_cache:
        #             # print("negative . . . ")
        #             score = lucene_perspective_pair_cache[(pId1, pId2)]
        #             if score > pp_threshold:
        #                 perspective_pair_variables.append(y)
        #                 perspective_pair_weights.append(7- 0.0001 * score)
        #                 # perspective_pair_variables_map[(pId1, pId2)] = score
        #                 prob += perspective_variables[i1] >= y
        #                 prob += perspective_variables[i2] >= y
        #                 prob += y >= (perspective_variables[i2] + perspective_variables[i1] - 1)
        #
        # total_obj += sum(x * obj for x, obj in zip(perspective_pair_variables, perspective_pair_weights))

        # print(perspective_pair_variables)

        # assert len(perspective_pair_variables) == len(perspective_pair_weights)

        # given perspectives, retrieve relevant evidences
        # evidence_variables = []
        # evidence_weights = []
        # evidence_ids = []
        # threshold = 10
        # perspective_to_evidence_variables = {}
        evidence_to_perspective_variables = {}
        # for p_text1, pId1, _ in perspective_given_claim:
        # perspective_to_evidence_variables[pId1] = []
        # lucene_evidences = get_evidence_from_pool(p_text1, 1)
        # for (eIdx, (e_text, eId, eScore)) in enumerate(lucene_evidences):
        # if pScore > threshold:
        #     x = LpVariable("pe" + str(pId1) + "-" + str(eId), 0, 1)
        # evidence_variables.append(x)
        # evidence_weights.append(pScore)
        # evidence_ids.append(eId)
        # perspective_to_evidence_variables[pId1].append((x, pScore, eId))
        # print(pScore)

        # if a perspective is active, it should be connected to at least one evidence
        # evidence is not active, unless it is connected to sth

        prob += total_obj
        status = prob.solve()
        print(LpStatus[status])

        # extract active variables
        used_evidences_and_texts = []
        persp_sup = []
        for pVar, p in zip(perspective_variables, perspective_given_claim_subset):
            pScore = p[2]
            p_text = p[0]
            # print(pVar)
            # print(value(pVar))
            if value(pVar) != None:
                if value(pVar) > 0.5:
                    lucene_evidences = get_evidence_from_pool(claim + p_text, 2)
                    evidences = []
                    if len(lucene_evidences) > 0:
                        (e_text, eId, eScore) = lucene_evidences[0]
                        evidences = eId
                        used_evidences_and_texts.append([eId, e_text.replace("`", "'")])
                    persp_sup.append((p[0], p[1], 1, [evidences], pScore))

            else:
                print("value is none")

        if vis_type == "graphical-viz":
            claim_persp_bundled = [(claim, persp_sup, [])]
        else:
            claim_persp_bundled = []

        persp_sup = [
            ([(7584, 'It will cause less re-offenders.'), (26958, 'Adequate punishment reduces future offenses.'),
              (26959, 'Just punishment will lead to less criminals re-offending. ')], [3, 0, 0, 0, 0],
             [367, 368, 2628, 2629, 7862, 6549])]
        persp_und = [([(7587, 'The onus should not be on punishing the criminal.'),
                       (26962, 'Punishment should not be the primary focus.'),
                       (26963, 'Our  main goal should not be punishing the criminal. ')], [0, 0, 0, 3, 0],
                      [7574, 7872])]

        context = {
            "claim_text": claim_text,
            "vis_type": vis_type,
            "perspectives_sorted": perspectives_sorted,
            "claim_persp_bundled": claim_persp_bundled,
            "used_evidences_and_texts": used_evidences_and_texts,
            # "claim": "",
            # "claim_id": claim_id,
            "persp_sup": persp_sup,
            "persp_und": persp_und,
        }

        print(context)

    else:
        context = {}

    return render(request, "vis_dataset_js_with_search_box.html", context)


###############################################
#     STEP 4 APIs
#     Topic
###############################################

def render_topic_annotation(request):
    username = request.user.username
    session = get_topic_hit_session(username)

    jobs = json.loads(session.jobs)
    finished = json.loads(session.finished_jobs)

    claims = [Claim.objects.get(id=cid) for cid in jobs]

    context = {
        'claims': claims
    }

    return render(request, "step4_topics/topic_interface.html", context)


def submit_topic_annotation(request):
    if request.method != 'POST':
        raise ValueError("submit_topic_annotation API only supports POST request")

    annos = json.loads(request.POST.get('annotations'))
    username = request.user.username
    session = get_topic_hit_session(username)

    # Update annotation in EquivalenceAnnotation table
    finished_cid_set = set()
    for anno in annos:
        cid = anno[0]
        label = anno[1]

        t = TopicAnnotation.objects.create(author=username, claim_id=cid, topics=label)
        t.save()

        if username != 'TEST':
            finished_cid_set.add(cid)

    for cid in finished_cid_set:
        c = Claim.objects.get(id=cid)
        c.topic_finished_counts += 1
        c.save()

    # increment duration in database
    delta = datetime.datetime.now(datetime.timezone.utc) - session.last_start_time
    session.duration = session.duration + delta
    session.job_complete = True
    session.save()

    res = HttpResponse(str(session.id), status=200)
    return res


def sunburst(request):
    # create a list of topics and populate their claim strings
    # claims = load_json(file_names["claim_annotation"])
    topics_to_claim = {}

    for c in claims[0:70]:
        topics = c['topics']
        claim_text = c['text']
        for topic_text in topics:
            if topic_text not in topics_to_claim:
                topics_to_claim[topic_text] = []
            topics_to_claim[topic_text].append(claim_text)

    data = []
    for key in topics_to_claim.keys():
        children = [{"name": x, "size": 1} for x in topics_to_claim[key]]
        data.append({
            "name": key,
            "children": children
        })

    context = {
        "data": json.dumps({
            "name": "topics",
            "children": data
        })
    }
    return render(request, "topics-sunburst/sunburst.html", context)


topics_map = {
    'culture': 'Culture',
    'society': 'Society',
    'world_international': 'World',
    'politics': 'Politics',
    'law': 'Law',
    'religion': 'Religion',
    'human_rights': 'Human Rights',
    'economy': 'Economy',
    'environment': 'Environment',
    'science_and_technology': 'Science',
    'education': 'education',
    'digital_freedom': 'Digital Freedom',
    'freedom_of_speech': 'F. of Speech',
    'health_and_medicine': 'Health',
    'gender': 'Gender',
    'ethics': 'Ethics',
    'sports_and_entertainments': 'Sports',
    'philosophy': 'Philosophy'
}


def sunburst(request):
    # create a list of topics and populate their claim strings
    # claims = load_json(file_names["claim_annotation"])
    topics_to_claim = {}

    for c in random.sample(claims, 55):
        topics = c['topics']
        claim_text = c['text']
        for topic_text in topics:
            if topic_text not in topics_to_claim:
                topics_to_claim[topic_text] = []
            topics_to_claim[topic_text].append(claim_text)

    print(topics_to_claim.keys())

    data = []
    total_childen = 0
    for key in topics_to_claim.keys():
        children = [{"name": x, "value": 1} for x in topics_to_claim[key]]
        if len(children) > 1:
            data.append({
                "name": topics_map[key],
                "children": children,
                "value": len(children)
            })
            total_childen += len(children)

    # the bottom half
    # data.append({
    #     "name": "dummy",
    #     "children": [{"name": "fake", "value": 1} for x in range(0,total_childen)]
    # })

    context = {
        "data": json.dumps([{
            "name": "",
            "children": data
        }])
    }
    return render(request, "topics-sunburst/sunburst2.html", context)


@login_required
def render_human_eval(request, claim_id):
    if claim_id not in claim_dict:
        return HttpResponse("Claim with id = {} doesn't exist in our database. ".format(claim_id), status=404)

    claim_title = claim_dict[claim_id]['text']

    cands = es.get_perspective_from_pool(claim_title, 50)

    context = {
        'claim_id': claim_id,
        'claim_title': claim_title,
        'persp_candidates': cands
    }
    return render(request, 'human_eval.html', context)


def retrieve_evidence_candidates(request, cid, pid):
    if (cid not in claim_dict) or (pid not in persp_dict):
        return HttpResponse("Claim with id = {} doesn't exist in our database. ".format(cid), status=404)

    claim_title = claim_dict[cid]['text']
    persp_title = persp_dict[pid]

    cands = get_evidence_from_pool(claim_title + '. ' + persp_title, 40)

    return JsonResponse({
        'evi_candidates': cands
    })


def normalize(num):
    return math.floor(num * 100) / 100.0


@login_required
@csrf_protect
def submit_human_anno(request):
    if request.method != 'POST':
        raise ValueError("submit_topic_annotation API only supports POST request")

    claim_id = request.POST.get('claim_id')
    annos = json.loads(request.POST.get('annotations'))
    username = request.user.username

    HumanAnnotation.objects.create(author=username, claim_id=claim_id, annotation=json.dumps(annos))

    return HttpResponse("Submission success!", status=200)


def render_demo(request):
    return render(request, "demo/demo_home.html", {})
