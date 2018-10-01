import json

from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render

from .util.helper import get_all_claim_title_id, get_claim_given_id
file_names = {
    "iDebate": '../data/idebate/idebate.json'
}


def load_json(file_name):
    with open(file_name, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
        return data


def get_json(request):
    data = load_json(file_names["iDebate"])
    return JsonResponse({"data": data})


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

