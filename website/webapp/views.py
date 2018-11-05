import json

from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect

from django.core.mail import send_mail, BadHeaderError
from django.shortcuts import render, redirect
from .forms import ContactForm

from webapp.models import *
from webapp.auth import get_hit_session
from webapp.util.step2.equivalence_auth import get_equivalence_hit_session
import datetime

file_names = {
    "iDebate": '../data/idebate/idebate.json'
}

"""Helper functions"""
def load_json(file_name):
    with open(file_name, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        return data


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


""" APIs """
def get_json(request):
    data = load_json(file_names["iDebate"])
    return JsonResponse({"data": data})


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
    return render(request, 'claim_neg_anno.html', {})


@login_required
def vis_relation(request, claim_id):
    try:
        claim = Claim.objects.get(id=claim_id)
    except Claim.DoesNotExist:
        pass  # TODO: Do something? 404?

    perspective_pool = get_pool_from_claim_id(claim_id)

    return render(request, 'claim_relation.html', {
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
        session = get_hit_session(username)

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
        session = get_hit_session(username)

        if claim_id and annos:
            for a in annos:
                parts = a.split(',')
                if len(parts) != 2:
                    return HttpResponse("Submission Failed! Annotation not valid.", status=400)

                persp_id = parts[0]
                rel = parts[1]
                anno_entry = PerspectiveRelation.objects.create(author=username, claim_id=claim_id, perspective_id=persp_id, rel=rel)
                anno_entry.save()

        else:
            return HttpResponse("Submission Failed! Annotation not valid.", status=400)

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
    session = get_hit_session(username)
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
    return render(request, "list_tasks.html", context)

def render_instructions(request):
    return render(request, "instructions.html", {})

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
    session = get_hit_session(username)
    session.last_start_time = datetime.datetime.now(datetime.timezone.utc)
    session.save()
    try:
        claim = Claim.objects.get(id=claim_id)
    except Claim.DoesNotExist:
        pass  # TODO: Do something? 404?

    perspective_pool = get_all_persp(claim_id)

    return render(request, 'normalize_persp.html', {
        "claim": claim,
        "perspective_pool": perspective_pool
    })


# Step 2 apis

@login_required
def vis_persp_equivalence(request, claim_id):
    username = request.user.username
    session = get_equivalence_hit_session(username)
    session.last_start_time = datetime.datetime.now(datetime.timezone.utc)
    session.save()

    try:
        claim = Claim.objects.get(id=claim_id)
    except Claim.DoesNotExist:
        pass  # TODO: Do something? 404?

    perspective_pool = get_all_persp(claim_id)

    candidates = {}

    for persp in perspective_pool:
        cand_ids = set(json.loads(Perspective.objects.get(id=persp.id).similar_persps))
        cand_persps = Perspective.objects.filter(id__in=cand_ids)
        candidates[persp.id] = cand_persps

    return render(request, 'step2/persp_equivalence.html', {
        "claim": claim,
        "perspective_pool": perspective_pool,
        "candidates": candidates
    })

def render_step2_instructions(request):
    return render(request, "step2/instructions.html", {})

def render_step2_task_list(request):
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

    return render(request, 'step2/task_list.html', context)


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
            c = Claim.objects.get(claim_id=claim_id)
            c.equivalence_finished_counts += 1
            c.save()

        return HttpResponse("Submission Success!", status=200)

@login_required
@csrf_protect
def step2_submit_instr(request):
    if request.method != 'POST':
        raise ValueError("submit_instr API only supports POST request")
    else:
        username = request.user.username
        session = get_equivalence_hit_session(username)

        session.instruction_complete = True
        session.save()
        return HttpResponse("Submission Success!", status=200)
