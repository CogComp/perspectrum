from elasticsearch import Elasticsearch
es = Elasticsearch(['http://bronte.cs.illinois.edu'],port=8080)

def get_top_perspectives(evidence):
    res = es.search(index="perspectivesandclaims", doc_type="text", body={"query": {"match": {"title": evidence}}}, size=50)
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


if __name__ == '__main__':
    # Create indices for google perspectives
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python ... [google_persp]", file=sys.stderr)
        exit(1)

    google_persp = sys.argv[1]
    with open(google_persp, 'r', encoding='utf-8') as fin:
        cands = json.load(fin)

    # Create indices
    createIndices("perspectivesandclaims_google", cands)

    print(get_top_google_perspectives("The use of child performers should be banned. Being a performer limits a child’s formal education")[:1])
