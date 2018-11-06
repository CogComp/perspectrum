import math
from allennlp.service.predictors import Predictor
from allennlp.models.archival import load_archive
from ccg_nlpy import remote_pipeline

def load_model():
    model = "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz"
    archive = load_archive(model)
    predictor = Predictor.from_archive(archive, 'textual-entailment')
    return predictor

stop_words = ["me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", ",", ".", "?"]

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    lst3_nostopword = [x for x in lst3 if x not in stop_words]
    return lst3_nostopword

def read_inputs():
    pipeline = remote_pipeline.RemotePipeline(server_api="http://macniece.seas.upenn.edu:4001")

    output_json = []
    import json
    file = "/home/danielk/perspectives/data/perspectives.json"

    with open(file, encoding='utf-8') as data_file:
        predictor = load_model()
        data = json.loads(data_file.read())

    # pre-process the lemmas
    for i, item1 in enumerate(data):
        try:
            doc = pipeline.doc(item1["title"])
            lemmas = [x["label"].lower() for x in doc.get_lemma]
            data[i]["lemmas"] = lemmas
        except ValueError:
            data[i]["lemmas"] = []
            print("Something happened . . . ")

    print("Done with pre-processing the lemmas . . . ")

    def save():
        with open("/home/danielk/perspectives/data/perspectives_pairs.json", 'a') as fp:
            json.dump(output_json, fp)

    for i, item1 in enumerate(data):
        print(f" - Processes {i} out of {len(data)}")
        # filter the perspectives
        filtered_list = []
        split1 = item1["lemmas"]
        for item2 in data:
            split2 = item2["lemmas"]
            shared = intersection(split1, split2)
            if(len(shared) > 1):
                filtered_list.append(item2)

        print(f"Size of selected list: {len(filtered_list)}")
        if i % 100 == 1:
            save()
        for item2 in filtered_list:
            # print("item2: " + str(item2["title"]))
            inputs = {
                "premise": str(item1["title"]),
                "hypothesis": str(item2["title"])
            }
            try:
                json = predictor.predict_json(inputs)
                label_logits = json["label_logits"]
                exps = [math.exp(x) for x in label_logits]
                sumexps = sum(exps)
                prob = [e / sumexps for e in exps]
                output_json.append({"id1": item1["id"], "id2": item2["id"], "prob": prob})
            except ValueError:
                print("Something happened . . . ")
    save()

def test_entailment():
    predictor = load_model()
    inputs = {
        "premise": "I always write unit tests for my code.",
        "hypothesis": "One time I didn't write any unit tests for my code."
    }
    json = predictor.predict_json(inputs)
    print(json)

def logit_to_prob(logit):
    odds = math.exp(logit)
    return odds / (1 + odds)

if __name__ == "__main__":
    # test_entailment()
    read_inputs()
    # test_entailment()

