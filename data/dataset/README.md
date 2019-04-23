# Perspective Dataset

### Dataset Format
There dataset consists of 5 json files
`perspectrum_with_answers_v1.0.json`: Contains all claims and all gold annotations (perspective clusters with their evidence paragraphs) to the claim.
```python
  {
    "cId": 0,
    "text": "according the internal human rights council ...",
    "source": "debatewise",
    "perspectives": [  # Gold perspectives
        {
            "pids": [2, 3, 4],  # First perspective cluster; perspectives within cluster is considered 'equivalent'
            "stance_label_3": "support", # coarse stance label set = {support, undermine, not-a-perspective}
            "stance_label_5": "mildly_support", # fine stance label set = {support, mildly-support, mildly-undermine, undermine, not-a-perspective}
            "voter_counts": [1, 3, 0, 0, 0],  # Crowdsourcing annotation counts with fine stance label set
            "evidence": [10, 20] # Evidence paragraph ids
        },
        {
            "pids": [101],  # Another perspective cluster
            "stance_label_3": "undermine",
            "stance_label_5": "undermine",
            "voter_counts": [0, 0, 0, 4, 0],
            "evidence": [10, 20]
        }
    ]
  }
```

`perspective_pool_v1.0.json`: Contains all perspectives
```json
  {
    "pId": 0,
    "text": "universal healthcare is a right",
    "source": "idebate"
  }

```

`evidence_pool_v1.0.json`: Contains all evidence paragraphs
```json
  {
    "eId": 0,
    "text": "according the internal human rights council ...",
    "source": "procon"
  }

```

`dataset_split_v1.0.json`: Contains the dataset split
```python
{
    "1": "train", # claim with id = 1 is in the training set
    "2": "train",
    "3": "test",
    ...
}
```
`topic.json`: Contains the topics (annotated by crowdsource workers) covered by each claim
```json
[
    {
        "claim_id": 1004,
        "topics": ["digital_freedom", "science_and_technology"]
    },
    {
        "claim_id": 1005,
        "topics": []
    }
]
```
