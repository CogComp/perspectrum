import json
import logging
import math
import numpy as np
import sys
import zipfile
from io import BytesIO

from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.core.mail import send_mail, BadHeaderError
from django.shortcuts import render, redirect
from django.core.files import File
from sklearn.cluster import DBSCAN
from django.contrib.auth import logout

# from experiment.query_elasticsearch import get_perspective_from_pool
# from experiment.query_elasticsearch import get_evidence_from_pool
from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, value, os

# from experiment.run_bert_on_perspectrum import BertBaseline

from webapp.models import *

import random
from copy import deepcopy

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


def render_login_page(request):
    """
    Renderer for login page
    """
    return render(request, "login.html", {})


def logout_request(request):
    logout(request)
    return render_login_page(request)



def successView(request):
    return HttpResponse('Success! Thank you for your message.')


# def bert_baseline(request, claim_text=""):
#     print(claim_text)
#     if claim_text != "":
#         claim = claim_text  #
#
#         prob = LpProblem("perspectiveOptimization", LpMaximize)
#
#         # given a claim, extract perspectives
#         perspective_given_claim = [(p_text, pId, pScore / len(p_text.split(" "))) for p_text, pId, pScore in
#                                    get_perspective_from_pool(claim, 30)]
#
#         # create binary variables per perspective
#         perspective_variables = []
#         perspective_weights = []
#         perspective_ids = []
#         perspective_given_claim_subset = []
#
#         for (pIdx, (p_text, pId, pScore)) in enumerate(perspective_given_claim):
#             if pScore > 1.55:
#                 x = LpVariable("p" + str(pIdx), 0, 1)
#                 perspective_variables.append(x)
#                 perspective_weights.append(pScore)
#                 perspective_ids.append(pId)
#                 perspective_given_claim_subset.append((p_text, pId, pScore))
#                 # print(pScore)
#
#         # print(len(perspective_variables))
#
#         total_obj = sum(x * obj for x, obj in zip(perspective_variables, perspective_weights))
#
#         # maximum and minimum number of perspectives selected
#         total_weight = sum(x * 1.0 for x in perspective_variables)
#         prob += total_weight <= 10
#         # prob += total_weight >= 5
#
#         assert (len(perspective_variables) == len(perspective_weights))
#
#         prob += total_obj
#         status = prob.solve()
#         print(LpStatus[status])
#
#         # extract active variables
#         used_evidences_and_texts = []
#         persp_sup = []
#         for pVar, p in zip(perspective_variables, perspective_given_claim_subset):
#             pScore = p[2]
#             p_text = p[0]
#             # print(pVar)
#             # print(value(pVar))
#             if value(pVar) != None:
#                 if value(pVar) > 0.5:
#                     lucene_evidences = get_evidence_from_pool(claim + p_text, 2)
#                     evidences = []
#                     if len(lucene_evidences) > 0:
#                         (e_text, eId, eScore) = lucene_evidences[0]
#                         evidences = eId
#                         used_evidences_and_texts.append([eId, e_text.replace("`", "'")])
#                     persp_sup.append((p[0], p[1], 1, [evidences], pScore))
#
#             else:
#                 print("value is none")
#
#         claim_persp_bundled = [(claim, persp_sup, [])]
#
#         context = {
#             "claim_persp_bundled": claim_persp_bundled,
#             "used_evidences_and_texts": used_evidences_and_texts
#         }
#     else:
#         context = {}
#
#     return render(request, "vis_dataset_js.html", context)


# ### loading the BERT solvers
# bb_relevance = BertBaseline(task_name="perspectrum_relevance",
#                             saved_model="model/relevance/perspectrum_relevance_lr2e-05_bs32_epoch-0.pth", no_cuda=True)
# bb_stance = BertBaseline(task_name="perspectrum_stance",
#                          saved_model="model/stance/perspectrum_stance_lr2e-05_bs16_epoch-4.pth", no_cuda=True)
# bb_equivalence = BertBaseline(task_name="perspectrum_equivalence",
#                               saved_model="model/equivalence/perspectrum_equivalence_lr3e-05_bs32_epoch-2.pth",
#                               no_cuda=True)
#
# logging.disable(sys.maxsize)  # Python 3


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
                                   get_perspective_from_pool(claim, 3)]

        perspective_relevance_score = bb_relevance.predict_batch(
            [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

        perspective_stance_score = bb_stance.predict_batch(
            [(claim, p_text) for (p_text, pId, _) in perspective_given_claim])

        perspectives_sorted = [(p_text, pId, normalize(luceneScore), normalize(perspective_relevance_score[i]),
                                normalize(perspective_stance_score[i])) for i, (p_text, pId, luceneScore) in
                               enumerate(perspective_given_claim)]

        perspectives_sorted = sorted(perspectives_sorted, key=lambda x: -x[3])

        similarity_score = np.zeros((len(perspective_given_claim), len(perspective_given_claim)))
        perspectives_equivalences = []
        for i, (p_text1, _, _) in enumerate(perspective_given_claim):
            list1 = []
            # list2 = []
            for j, (p_text2, _, _) in enumerate(perspective_given_claim):
                # if i != j:
                list1.append((claim + " . " + p_text1, p_text2))
                # list2.append((claim + " . " + p_text2, p_text1))

            predictions1 = bb_equivalence.predict_batch(list1)
            # predictions2 = bb_equivalence.predict_batch(list2)

            for j, (p_text2, _, _) in enumerate(perspective_given_claim):
                if i != j:
                    perspectives_equivalences.append((p_text1, p_text2, predictions1[j], predictions1[j]))
                    similarity_score[i, j] = predictions1[j]
                    similarity_score[j, i] = predictions1[j]

        distance_scores = -similarity_score

        # rescale distance score to [0, 1]
        distance_scores -= np.min(distance_scores)
        distance_scores /= np.max(distance_scores)

        clustering = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_scores)
        print(cluster_labels)
        max_val = max(cluster_labels)
        for i, _ in enumerate(cluster_labels):
            max_val += 1
            if cluster_labels[i] == -1:
                cluster_labels[i] = max_val

        persp_sup = []
        persp_sup_flash = []
        persp_und = []
        persp_und_flash = []

        perspective_clusters = {}
        for i, (p_text, pId, luceneScore, relevance_score, stance_score) in enumerate(perspectives_sorted):
            if relevance_score > 0.0:
                id = cluster_labels[i]
                if id not in perspective_clusters:
                    perspective_clusters[id] = []
                perspective_clusters[id].append((p_text, pId, stance_score))

        for cluster_id in perspective_clusters.keys():
            stance_list = []
            perspectives = []
            persp_flash_tmp = []
            for (p_text, pId, stance_score) in perspective_clusters[cluster_id]:
                stance_list.append(stance_score)
                perspectives.append((pId, p_text))
                persp_flash_tmp.append((p_text, pId, cluster_id + 1, [], stance_score))
                # persp_sup.append((p[0], p[1], 1, [evidences], pScore))

            avg_stance = sum(stance_list) / len(stance_list)
            if avg_stance > 0.0:
                persp_sup.append((perspectives, [avg_stance, 0, 0, 0, 0], []))
                persp_sup_flash.extend(persp_flash_tmp)
            else:
                persp_und.append((perspectives, [avg_stance, 0, 0, 0, 0], []))
                persp_und_flash.extend(persp_flash_tmp)

        # if vis_type == "graphical-viz":
        #     claim_persp_bundled = [(claim, persp_sup, [])]
        # else:
        #     claim_persp_bundled = []

        claim_persp_bundled = [(claim, persp_sup_flash, persp_und_flash)]

        # persp_sup = [
        #     ([(7584, 'It will cause less re-offenders.'), (26958, 'Adequate punishment reduces future offenses.'),
        #       (26959, 'Just punishment will lead to less criminals re-offending. ')], [3, 0, 0, 0, 0],
        #      [367, 368, 2628, 2629, 7862, 6549])]
        # persp_und = [([(7587, 'The onus should not be on punishing the criminal.'),
        #                (26962, 'Punishment should not be the primary focus.'),
        #                (26963, 'Our  main goal should not be punishing the criminal. ')], [0, 0, 0, 3, 0],
        #               [7574, 7872])]

        context = {
            "claim_text": claim_text,
            "vis_type": vis_type,
            "perspectives_sorted": perspectives_sorted,
            "perspectives_equivalences": perspectives_equivalences,
            "claim_persp_bundled": claim_persp_bundled,
            "used_evidences_and_texts": [],  # used_evidences_and_texts,
            # "claim": "",
            # "claim_id": claim_id,
            "persp_sup": persp_sup,
            "persp_und": persp_und,
        }

        print(context)

    else:
        context = {}

    return render(request, "vis_dataset_js_with_search_box.html", context)



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



def render_demo(request):
    return render(request, "vis_dataset_js_with_search_box.html", {})
