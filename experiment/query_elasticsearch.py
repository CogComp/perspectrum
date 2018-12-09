from elasticsearch import Elasticsearch
import json
es = Elasticsearch(['http://bronte.cs.illinois.edu'],port=8080)

def get_top_perspectives(evidence):
    res = es.search(index="perspectivesandclaims", doc_type="text", body={"query": {"match": {"title": evidence}}}, size=500)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        id = doc['_source']["id"]
        score = doc['_score']
        perspective_text = doc['_source']["title"]
        output.append((perspective_text, id, score))
    # print(output)
    # print(len(output))
    return output

def get_top_evidences(perpsective):
    res = es.search(index="evidences", doc_type="text", body={"query": {"match": {"content": perpsective}}}, size=50)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        id = doc['_source']["id"]
        score = doc['_score']
        evidence_text = doc['_source']["content"]
        output.append((evidence_text, id, score))
    return output


def get_top_google_perspectives(text):
    res = es.search(index="perspectivesandclaims_google", doc_type="text", body={"query": {"match": {"candidate": text}}}, size=50)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        id = doc['_source']["perspective_id"]
        score = doc['_score']
        perspective_text = doc['_source']["candidate"]
        output.append((perspective_text, id, score))

    return output

def createIndices(indices_name, data):
    if es.indices.exists(indices_name):
        print("Index {} already exists... Aborting!" .format(indices_name))
        return

    es.indices.create(indices_name)
    for idx, doc in enumerate(data):    
        print("Processing id: {}".format(idx))
        es.index(index=indices_name, doc_type='text', id=idx, body=doc)

def get_top_re_step1_perspectives(text, num_cands=20):
    res = es.search(index="re_step1_claim_persp_high_quality", doc_type="text", body={"query": {"match": {"concat_title": text}}}, size=num_cands)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        cid = doc['_source']["claim_id"]
        pid = doc['_source']["perspective_id"]
        score = doc['_score']
        perspective_text = doc['_source']["concat_title"]
        output.append((perspective_text, cid, pid, score))

    return output

def get_top_sentences(text):
    res = es.search(index="sentence_pool", doc_type="text", body={"query": {"match": {"sent": text}}}, size=120)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        # id = doc['_source']["perspective_id"]
        score = doc['_score']
        perspective_text = doc['_source']["sent"]
        output.append((perspective_text, score))

    return output

def get_claims(text):
    res = es.search(index="claim_with_paraphrases", doc_type="text", body={"query": {"match": {"concat": text}}}, size=10)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        # id = doc['_source']["perspective_id"]
        score = doc['_score']
        concat = doc['_source']["concat"]
        id = doc['_source']["claim_id"]
        claim_title = doc['_source']["claim_title"]
        output.append((concat, claim_title, id, score))

    return output

def get_claims_with_claim_surface(text):
    res = es.search(index="claim_with_paraphrases", doc_type="text", body={"query": {"match": {"claim_title": text}}}, size=10)
    # print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        # id = doc['_source']["perspective_id"]
        score = doc['_score']
        concat = doc['_source']["concat"]
        id = doc['_source']["claim_id"]
        claim_title = doc['_source']["claim_title"]
        output.append((concat, claim_title, id, score))
    return output

def get_perspective_from_pool(text, size):
    res = es.search(index="perspective_pool_v0.2", doc_type="text", body={"query": {"match": {"text": text}}}, size=size)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        pId = doc['_source']["pId"]
        output.append((text, pId, score))
    return output

def get_evidence_from_pool(text, size):
    res = es.search(index="evidence_pool_v0.2", doc_type="text", body={"query": {"match": {"text": text}}}, size=size)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        eId = doc['_source']["eId"]
        output.append((text, eId, score))
    return output

if __name__ == '__main__':
    # print(json.dumps(data))
    data = get_top_sentences("He explains that community organizers have to build structures to hold")
    print(json.dumps(data))