import json

from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from .models import *

from .util.helper import get_all_claim_title_id, get_claim_given_id
file_names = {
    "iDebate": '../data/idebate/idebate.json'
}

"""Helper functions"""
def load_json(file_name):
    with open(file_name, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        return data

def get_json(request):
    data = load_json(file_names["iDebate"])
    return JsonResponse({"data": data})


def get_pool_from_claim_id(claim_id):
    """
    TODO: Change this function! Right now it's only for testing purpose
    :param claim_id: id of the claim
    :return:
    """
    related_persp_anno = RelationAnnotation.objects.filter(author=RelationAnnotation.GOLD, claim_id=claim_id).\
        order_by("?")[:2]
    related_persps = [Perspective.objects.get(id=rel.perspective_id) for rel in related_persp_anno]

    return related_persps


""" APIs """
def main_page(request):
    context = {
        "datasets": list(file_names.keys())
    }
    return render(request, 'main.html', context)

def vis_claims(request):
    data = load_json(file_names["iDebate"])
    claim_titles = get_all_claim_title_id(data)

    # print(len(claim_titles))
    context = {
        "claim_titles": claim_titles
    }
    return render(request, 'claims.html', context)


def vis_persps(request, claim_id):
    data = load_json(file_names["iDebate"])
    claim = get_claim_given_id(data, claim_id)
    context = {
        "claim": claim
    }

    return render(request, 'persp.html', context)


def vis_neg_anno(request, claim_id):
    return render(request, 'claim_neg_anno.html', {})


def vis_relation(request, claim_id):
    try:
        claim = Claim.objects.get(id=claim_id)
    except Claim.DoesNotExist:
        pass  # TODO: Do something?

    perspective_pool = get_pool_from_claim_id(claim_id)

    return render(request, 'claim_relation.html', {
        "perspective_pool": perspective_pool
    })

