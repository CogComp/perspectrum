import json

from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect

from django.core.mail import send_mail, BadHeaderError
from django.shortcuts import render, redirect
from .forms import ContactForm

from webapp.models import *
from webapp.util.step1.persp_verification_auth import get_persp_hit_session
from webapp.util.step2b.equivalence_auth import get_equivalence_hit_session
from webapp.util.step2a.paraphrase_auth import get_paraphrase_hit_session
from webapp.util.step3.evidence_auth import get_evidence_hit_session

from django.core.files import File

from collections import OrderedDict
import datetime
import random

file_names = {
    'perspective': 'data/dataset/perspective_pool_v0.1.json',
    'claim_annotation': 'data/dataset/perspectrum_with_answers_v0.1.json'
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
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id).\
        order_by("?")[:2]
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


def get_all_persp(claim_id):
    """
    :param claim_id: id of the claim
    :return: list of perspectives
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id).\
        order_by("?")
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


def get_all_google_persp(claim_id):
    """
    :param claim_id: id of the claim
    :return: list of perspectives
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD,
                                                            claim_id=claim_id, comment="google")\
        .order_by("?")
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


def get_all_original_persp(claim_id):
    """
    :param claim_id: id of the claim
    :return: list of perspectives
    """
    related_persp_anno = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id)\
        .exclude(comment="google").order_by("?")
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


""" APIs """
def personality(request):
    context = { }
    return render(request, 'personality.html', context)

@login_required
def main_page(request):
    context = {
        "datasets": list(file_names.keys())
    }
    return render(request, 'main.html', context)


@login_required
def vis_claims(request):
    claim_titles = []
    for c in Claim.objects.all():
        claim_titles.append((c.title, c.id))

    context = {
        "claim_titles": claim_titles
    }
    return render(request, 'claims.html', context)

@login_required
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

@login_required
def vis_spectrum_js(request, claim_id):
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

    return render(request, 'vis_dataset_js.html', context)

STANCE_FLIP_MAPPING = {
    "SUPPORT": "UNDERMINE",
    "MILDLY_SUPPORT": "MILDLY_UNDERMINE",
    "MILDLY_UNDERMINE": "MILDLY_SUPPORT",
    "UNDERMINE": "SUPPORT",
}

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
        claim_to['perspectives'].append(claim_from['perspectives'][c1_idx])
        if flip_stance:
            lbl3 = claim_to['perspectives'][-1]['stance_label_3']
            if lbl3 in STANCE_FLIP_MAPPING:
                claim_to['perspectives'][-1]['stance_label_3'] = STANCE_FLIP_MAPPING[lbl3]

            lbl5 = claim_to['perspectives'][-1]['stance_label_5']
            if lbl5 in STANCE_FLIP_MAPPING:
                claim_to['perspectives'][-1]['stance_label_5'] = STANCE_FLIP_MAPPING[lbl5]

            claim_to['perspectives'][-1]['voter_counts'].reverse()

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
            del persp["pids"][persp["pids"].index(perspective_id)]

    return HttpResponse("Success", status=200)


def merge_perspectives(request, cid1, pid1, cid2, pid2):

    if (cid1 == cid2) and (pid1 == pid2):
        return HttpResponse("Success", status=200)

    claim1 = claim_dict[cid1]
    claim2 = claim_dict[cid2]

    c1_idx = None
    for idx, p in enumerate(claim1['perspectives']):
        if pid1 in p['pids']:
            c1_idx = idx

    c2_idx = None
    for idx, p in enumerate(claim2['perspectives']):
        if pid2 in p['pids']:
            c2_idx = idx

    print(c1_idx, c2_idx)
    if (c1_idx != None) and (c2_idx != None):
        print(pid1, pid2)
        merged_pid = list(set(claim1['perspectives'][c1_idx]['pids'] + claim2['perspectives'][c2_idx]['pids']))
        merged_evi_ids = list(set(claim1['perspectives'][c1_idx]['evidence'] + claim2['perspectives'][c2_idx]['evidence']))
        claim1['perspectives'][c1_idx]['pids'] = merged_pid
        claim2['perspectives'][c2_idx]['pids'] = merged_pid
        claim1['perspectives'][c1_idx]['evidence'] = merged_evi_ids
        claim2['perspectives'][c2_idx]['evidence'] = merged_evi_ids

    if cid1 == cid2:
        del claim2['perspectives'][c2_idx]

    return HttpResponse("Success", status=200)

def save_updated_claim_on_disk(request, file_name):
    # convert map to list before saving
    claims_local = [claim_dict[k] for k in claim_dict.keys()]
    save_json(claims_local, "data/dataset/" + file_name)
    return HttpResponse("Success", status=200)

    return HttpResponse("Save Success", status=200)
persps = load_json(file_names["perspective"])
claims = load_json(file_names["claim_annotation"])
persp_dict = {}
claim_dict = {}
for p in persps:
    persp_dict[p["pId"]] = p["text"]

for c in claims:
    claim_dict[c["cId"]] = c

@login_required
def vis_dataset_side_by_side(request, claim_id1, claim_id2):
    persps = load_json(file_names["perspective"])
    claims = load_json(file_names["claim_annotation"])

    # claim_id1 = 300
    claim_id1 = int(claim_id1)

    c_title1 = claim_dict[claim_id1]["text"]
    persp_sup1 = []
    persp_und1 = []
    for cluster in claim_dict[claim_id1]["perspectives"]:
        titles = [str(pid) + ": " + persp_dict[pid] for pid in cluster["pids"]]

        if cluster['stance_label_3'] == "SUPPORT":
            persp_sup1.append(titles)
        elif cluster['stance_label_3'] == "UNDERMINE":
            persp_und1.append(titles)

    claim_id2 = int(claim_id2)

    c_title2 = claim_dict[claim_id2]["text"]
    persp_sup2 = []
    persp_und2 = []
    for cluster in claim_dict[claim_id2]["perspectives"]:
        titles = [str(pid) + ": " + persp_dict[pid] for pid in cluster["pids"]]

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

@login_required
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



@login_required
def vis_persps(request, claim_id):
    claim = Claim.objects.get(id=claim_id)
    rel_set = PerspectiveRelation.objects.filter(author=PerspectiveRelation.GOLD, claim_id=claim_id)
    rel_sup_ids = [r["perspective_id"] for r in rel_set.filter(rel='S').values("perspective_id")]
    rel_und_ids = [r["perspective_id"] for r in rel_set.filter(rel='U').values("perspective_id")]

    persp_sup = Perspective.objects.filter(id__in=rel_sup_ids)
    persp_und = Perspective.objects.filter(id__in=rel_und_ids)

    context = {
        "claim": claim,
        "persp_sup": persp_sup,
        "persp_und": persp_und,
    }

    return render(request, 'persp.html', context)


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
        cid = PerspectiveRelation.objects.filter(author="GOLD").exclude(comment="google").get(perspective_id=pid).claim_id
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
            user_para_str = p.user_generated.replace('"','\\"').replace('[\'', '["').replace('\']', '"]').replace('\', \'', '", "')\
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

    valid_persp_ids = ReStep1Results.objects.filter(label_3__in=["S", "U"], p_i_3__gt=0.5).values_list('perspective_id').distinct()

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

def render_step4_task_list(request):
    return render(request, "step4_topics/topic_interface.html", {})

evidence_label_mapping = {
    "sup" : "S",
    "nsup" : "N"
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
