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

if __name__ == '__main__':
    # data = get_top_re_step1_perspectives("The content of public speech is informed as much by the ideas and convictions of individuals engaged in free expression as it is by the concurrent acts of expression engaged in by other individuals. Free speech is a product of society and the processes driving the development and growth of society. The environment in which free speech is currently exercised is characterised by pervasive acts of expression – television commercials, billboards, spam email and advertisements on social media sites. Each of these forms of media is aimed at influencing opinions and behaviours. Active engagement with a book or a movie is often a prerequisite if an individual is to be influenced by its content.. The audience for the content contained in an advert does not necessarily choose to engage with its message. As a result of this, adverts are uniquely placed to bring issues and perspectives to the attention of individuals who might otherwise have been unaware of them. Advertising is a powerful political tool. For this reason the manner in which political causes can be advertised and the amount of funding spent on those adverts is, almost without exception, strictly regulated in most liberal democracies. Commercial content carried by for-profit organisations such as newspapers and television channels is expensive. The prominence of a message is affected by the amount of money that can be spent on increasing its length, rebroadcasting it and showing it to new audiences. When it comes to political speech, spending money is the best way to increase the efficacy and persuasiveness of a message. Irrespective of the qualities of a particular campaign, the qualifications of its candidates or the evidence underlying its policy proposals, its effectiveness will still be measured in the amount of money that it is able to spend on advertising. Legal restrictions on political spending are intended to prevent political speech from becoming a battle of budget rather than ideas – campaign finance laws are designed to protect the integrity, quality and efficacy of speech. In the USA the Bi-partisan Campaign Reform Act achieved this goal by preventing corporations from funding “electioneering communications” within 30 days of a caucus or 60 days of a general election. The Act prevented interest groups indirectly affiliated with particular candidates from spending money to support a candidates’ message. Although there are limits on the income that a politician can directly receive from donors, different rules apply to organisations that are not directly affiliated with that politician. And although a politician may receive criticism for receiving corporate money, corporations can contribute to causes indirectly, by providing funds of issue groups.", num_cands=2000)
    # print(json.dumps(data))
    data = get_top_sentences("He explains that community organizers have to build structures to hold")
    print(json.dumps(data))