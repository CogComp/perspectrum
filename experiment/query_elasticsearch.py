from elasticsearch import Elasticsearch
es = Elasticsearch(['http://bronte.cs.illinois.edu'],port=8080)

def get_top_perspectives(evidence):
    res = es.search(index="perspectivesandclaims", doc_type="text", body={"query": {"match": {"title": evidence}}}, size=50)
    print("%d documents found:" % res['hits']['total'])
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
    print("%d documents found:" % res['hits']['total'])
    output = []
    for doc in res['hits']['hits']:
        id = doc['_source']["id"]
        score = doc['_score']
        evidence_text = doc['_source']["content"]
        output.append((evidence_text, id, score))
    return output

if __name__ == '__main__':
    # get_top_perspectives("Spending so much time either performing or training limits the amount of formal education the child can receive. For example, in the UK and other countries, child performers are only required to be educated for three hours each day. [1] Additionally, the focus on the specialised skill of the child (e.g., acting, dancing, etc.) may detract from their family\u2019s or their own interest in formal education.")
    get_top_evidences("The use of child performers should be banned. Being a performer limits a child’s formal education")