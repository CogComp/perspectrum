import json

from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render

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
    claim_titles = []
    for x in data:
        claim_titles.append(x["claim_title"])

    # print(len(claim_titles))
    context = {
        "claim_titles": claim_titles
    }
    return render(request, 'claims.html', context)
