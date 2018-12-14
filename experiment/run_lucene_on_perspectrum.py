import json
import random

from query_elasticsearch import get_perspective_from_pool
from query_elasticsearch import get_evidence_from_pool

topics_to_claim = {}

def load_json(file_name):
    with open(file_name) as data_file:
        data = json.loads(data_file.read())
        return data

gold_claims = load_json('/Users/daniel/ideaProjects/perspective/data/dataset/perspectrum_with_answers_v0.2.json')
gold_perspectives = load_json('/Users/daniel/ideaProjects/perspective/data/dataset/perspective_pool_v0.2.json')
gold_evidences = load_json('/Users/daniel/ideaProjects/perspective/data/dataset/evidence_pool_v0.2.json')
dataset_split = load_json('/Users/daniel/ideaProjects/perspective/data/dataset/dataset_split_v0.2.json')
train_ids = [int(id) for id, cat in dataset_split.items() if str(cat) == "train"]
test_ids = [int(id) for id, cat in dataset_split.items() if str(cat) == "test"]
dev_ids = [int(id) for id, cat in dataset_split.items() if str(cat) == "dev"]
gold_claims_train = [c for c in gold_claims if int(c['cId']) in train_ids]
gold_claims_test = [c for c in gold_claims if c['cId'] in test_ids]
gold_claims_dev = [c for c in gold_claims if c['cId'] in dev_ids]

gold_persp_dict = {}
gold_ev_dict = {}
for p in gold_perspectives:
    gold_persp_dict[p["pId"]] = p["text"]

for e in gold_evidences:
    gold_ev_dict[e["eId"]] = e

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def cache_lucene():
    # for any claim, retrieve the relevant perspectives
    lucene_annotation_cache = {}

    for c in gold_claims:
        claim_text = c['text']
        cId = c['cId']
        print(cId)
        lucene_annotation_cache[cId] = {}
        lucene_annotation_cache[cId]["p_given_c"] = get_perspective_from_pool(claim_text, 50)

        relevant_perspectives = []
        for p in c["perspectives"]:
            relevant_perspectives.extend(p["pids"])

        print(relevant_perspectives)

        lucene_annotation_cache[cId]["p_given_pc"] = {}
        for pId in set(relevant_perspectives):
            # print(gold_persp_dict[pId])
            pText = gold_persp_dict[pId]
            lucene_annotation_cache[cId]["p_given_pc"][pId] = get_perspective_from_pool(claim_text + " " + pText, 30)

        lucene_annotation_cache[cId]["e_given_pc"] = {}
        for pId in set(relevant_perspectives):
            pText = gold_persp_dict[pId]
            lucene_annotation_cache[cId]["e_given_pc"][pId] = get_evidence_from_pool(claim_text + " " + pText, 30)

    import json
    with open('/Users/daniel/ideaProjects/perspective/data/lucene_cache/cache_all.json', 'w') as outfile:
        json.dump(lucene_annotation_cache, outfile)


lucene_annotation_cache = load_json('/Users/daniel/ideaProjects/perspective/data/lucene_cache/cache_all.json')

train_claims = []

def relevant_perspectives():
    # for any claim, retrieve the relevant perspectives

    def eval(claim_list):
        for topK in range(2, 60, 1):
            recall_list = []
            precision_list = []
            for c in claim_list:
                cId = str(c['cId'])
                # print(cId)
                lucene_perspectives = [pId for (p_text, pId, pScore) in lucene_annotation_cache[cId]["p_given_c"]][0:topK]

                # gold_relevant_perspectives = []
                # for p in c["perspectives"]:
                    # gold_relevant_perspectives.extend(p["pids"])

                # inter = intersection(gold_relevant_perspectives, lucene_perspectives)

                inter = 0
                for lucenePId in lucene_perspectives:
                    for p in c["perspectives"]:
                        if lucenePId in p["pids"]:
                            inter += 1
                            break

                # recall = 1.0 * len(inter) / len(gold_relevant_perspectives)
                recall = 1.0 * inter / len(c["perspectives"])
                precision = 1.0 * inter / len(lucene_perspectives)

                recall_list.append(recall)
                precision_list.append(precision)

            overall_recall = sum(recall_list) / len(recall_list)
            overall_precision = sum(precision_list) / len(precision_list)
            f1 = 2 * overall_recall * overall_precision / (overall_precision + overall_recall)
            print(str(topK) + "\t" + str(overall_recall) + "\t" + str(overall_precision) + "\t" + str(f1))

    print("train + dev... ")
    eval(gold_claims_train + gold_claims_dev)

    print("test")
    eval(gold_claims_test)

def perspective_stances_counts():
    def eval(claim_list):
        support_count = 0
        undermine_count = 0
        for c in claim_list:
            for p in c["perspectives"]:
                stance = str(p['stance_label_3'])
                if stance == "SUPPORT":
                    support_count += 1
                if stance == "UNDERMINE":
                    undermine_count += 1

        print("Support count: " + str(support_count))
        print("Undermine count: " + str(undermine_count))

    print("train + dev... ")
    eval(gold_claims_train + gold_claims_dev)

    print("test")
    eval(gold_claims_test)

# since this does not make sense for lucene, didn't implement it
def perspective_stances():
    pass

def perspective_equivalence_static_baselines():
    def eval(claim_list):
        positive_list = []
        negative_list = []
        for c in claim_list:
            positive_pairs = []
            negative_pairs = []
            all_ids = []
            for p in c["perspectives"]:
                all_ids.extend(p["pids"])
                for pid1 in p["pids"]:
                    for pid2 in p["pids"]:
                        positive_pairs.append((pid1, pid2))
                        positive_pairs.append((pid2, pid1))
            for pid1 in all_ids:
                for pid2 in all_ids:
                    if (pid1, pid2) not in positive_pairs:
                        negative_pairs.append((pid1, pid2))
                        negative_pairs.append((pid2, pid1))

            positive_list.append(1.0 * len(positive_pairs) / (len(positive_pairs) + len(negative_pairs)))
            negative_list.append(1.0 * len(negative_pairs) / (len(positive_pairs) + len(negative_pairs)))

        print(sum(positive_list) / len(positive_list))
        print(sum(negative_list) / len(negative_list))

    print("train + dev... ")
    eval(gold_claims_train + gold_claims_dev)

    print("test")
    eval(gold_claims_test)

import numpy as np

def perspective_equivalence():
    # for any claim, retrieve the relevant perspectives
    def eval(claim_list):
        for threshold in np.arange(0.0, 5, 0.1):
            recall_list = []
            precision_list = []
            for c in claim_list:
                relevant_perspectives = []
                for p in c["perspectives"]:
                    relevant_perspectives.extend(p["pids"])
                # print(relevant_perspectives)
                tp = 0
                fp = 0
                fn = 0
                cId = str(c['cId'])
                claim_text = c["text"]

                positive_pairs = []
                all_ids = []
                for p in c["perspectives"]:
                    all_ids.extend(p["pids"])
                    for pid1 in p["pids"]:
                        for pid2 in p["pids"]:
                            positive_pairs.append((pid1, pid2))
                            positive_pairs.append((pid2, pid1))

                lucene_scores = {}

                for pId1 in relevant_perspectives:
                    # print("------")
                    pText1 = gold_persp_dict[pId1]
                    # print(lucene_annotation_cache[cId]["p_given_pc"].keys())
                    # print(unicode(pId1))
                    if unicode(pId1) in lucene_annotation_cache[cId]["p_given_pc"]:
                        for (p_text, pId2, pScore) in lucene_annotation_cache[cId]["p_given_pc"][unicode(pId1)]:
                            lucene_scores[(pId1, pId2)] = 1.0 * pScore / (len(claim_text.split(" ")) + len(pText1.split(" ")))

                for pid1 in all_ids:
                    for pid2 in all_ids:

                        if pid1 >= pid2:
                            continue

                        score = []
                        if (pid1, pid2) in lucene_scores:
                            score.append(lucene_scores[(pid1, pid2)])
                        if (pid2, pid1) in lucene_scores:
                            score.append(lucene_scores[(pid2, pid1)])

                        if len(score) > 0:
                            score = sum(score) / len(score)
                        else:
                            score = 0.0

                        if score >= threshold and (pid1, pid2) in positive_pairs:
                            tp += 1
                        if score < threshold and (pid1, pid2) in positive_pairs:
                            fn += 1
                        if score >= threshold and (pid1, pid2) not in positive_pairs:
                            fp += 1

                if tp + fp > 0:
                    precision_list.append(tp / (tp + fp))
                elif tp == 0:
                    precision_list.append(1.0)
                else:
                    precision_list.append(0.0)

                if tp + fn > 0:
                    recall_list.append(tp / (tp + fn))
                elif tp == 0:
                    recall_list.append(1.0)
                else:
                    recall_list.append(0.0)

            overall_recall = sum(recall_list) / len(recall_list)
            overall_precision = sum(precision_list) / len(precision_list)
            f1 = 2 * overall_recall * overall_precision / (overall_precision + overall_recall)
            print(str(threshold) + "\t" + str(overall_recall) + "\t" + str(overall_precision) + "\t" + str(f1))

    print("train + dev... ")
    eval(gold_claims_train + gold_claims_dev)

    print("test")
    eval(gold_claims_test)

def supporting_evidences():
    def eval(claim_list):
        for threshold in np.arange(0.0, 4.0, 0.1):
            recall_list = []
            precision_list = []
            length_list = []
            for c in claim_list:
                cId = str(c['cId'])
                claim_text = c["text"]
                for p in c["perspectives"]:
                    gold_evidences = p["evidence"]
                    selected_evidences = []
                    for pid1 in p["pids"]:
                        pText1 = gold_persp_dict[pid1]
                        for (e_text, eId2, eScore) in lucene_annotation_cache[cId]["e_given_pc"][unicode(pid1)]:
                            normalized_score = 1.0 * eScore / (len(claim_text.split(" ")) + len(pText1.split(" ")))
                            if normalized_score >= threshold:
                                selected_evidences.append(eId2)

                        inter = len(set(intersection(selected_evidences, gold_evidences)))

                        # recall = 1.0 * len(inter) / len(gold_relevant_perspectives)
                        recall = 1.0 * inter / len(gold_evidences) if len(gold_evidences) > 0 else 1.0
                        precision = 1.0 * inter / len(selected_evidences) if len(selected_evidences) > 0 else 1.0
                        recall_list.append(recall)
                        precision_list.append(precision)
                        length_list.append(len(set(selected_evidences)))

            overall_recall = 1.0 * sum(recall_list) / len(recall_list)
            overall_precision = 1.0 * sum(precision_list) / len(precision_list)
            overall_length = 1.0 * sum(length_list) / len(length_list)
            f1 = 2 * overall_recall * overall_precision / (overall_precision + overall_recall)
            print(str(threshold) + "\t" + str(overall_recall) + "\t" + str(overall_precision) + "\t" + str(f1) + "\t" + str(overall_length) )

    print("train + dev... ")
    eval(gold_claims_train + gold_claims_dev)

    print("test")
    eval(gold_claims_test)

def print_lucene_evidences_on_disk():
    def eval(claim_list):
        with open('evidences.tsv', 'a', encoding='utf-8') as the_file:
            # the_file.write('Hello\n')
        # for threshold in np.arange(0.0, 4.0, 0.1):
        #     recall_list = []
        #     precision_list = []
        #     length_list = []
            for c in random.sample(claim_list, 20):
                cId = str(c['cId'])
                claim_text = c["text"]
                for p in random.sample(c["perspectives"], 1):
                    # gold_evidences = p["evidence"]
                    # selected_evidences = []
                    for pid1 in p["pids"]:
                        pText1 = gold_persp_dict[pid1]
                        # print(pid1)
                        # print(lucene_annotation_cache[cId]["e_given_pc"].keys())
                        for (e_text, eId2, eScore) in lucene_annotation_cache[cId]["e_given_pc"][str(pid1)]:
                            eId2 = str(int(eId2))
                            pid1 = str(int(pid1))
                            # print(cId)
                            # print(pid1)
                            # print(eId2)
                            the_file.write(cId + '\t' + str(claim_text) + '\t' + pid1 + '\t' + str(pText1) + '\t' + eId2  + '\t' + str(e_text) + '\n')

    # print("train + dev... ")
    # eval(gold_claims_train + gold_claims_dev)

    print("test")
    eval(gold_claims_test)



# def experiments():
#     data_dir = "/shared/shelley/khashab2/perspective/data/dataset/perspective_stances/"
#     data_dir_output = data_dir + "output/"
    # train_and_test(data_dir=data_dir, do_train=True, do_eval=True, output_dir=data_dir_output,task_name="Mrpc")

# def evaluation_with_pretrained():
#     bert_model = "/shared/shelley/khashab2/perspective/data/dataset/perspective_stances/output/output.pth"
#     data_dir = "/shared/shelley/khashab2/perspective/data/dataset/perspective_stances/"
#     data_dir_output = data_dir + "output2/"
    # train_and_test(data_dir=data_dir, do_train=False, do_eval=True, output_dir=data_dir_output,task_name="Mrpc",saved_model=bert_model)

if __name__ == "__main__":
    # cache_lucene()
    # relevant_perspectives()
    # perspective_stances_counts()
    # perspective_equivalence_static_baselines()
    # perspective_equivalence()
    # supporting_evidences()
    print_lucene_evidences_on_disk()