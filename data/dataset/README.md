# Perspective Dataset

## 11.26.2018 Update 
### Task description
Given a claim and a pool of perspectives, find all perspectives that either support or undermine the claim.
### Dataset Format
3 json files 
- perspective_pool_v1.0.json
- evidence_pool_v1.0.json
- perspectrum_annotations_v1.0.json


`perspective_pool_v1.0.json`:
```json
  {
    "pId": 0,
    "text": "universal healthcare is a right",
    "source": "idebate"
  }

```

`evidence_pool_v1.0.json`:
```json
  {
    "eId": 0,
    "text": "according the internal human rights council ...",
    "source": "procon"
  }

```

`perspectrum_with_answers_v1.0.json`: 
```python
  {
    "cId": 0,
    "text": "according the internal human rights council ...",
    "source": "debatewise",
    "perspectives": [
      {
        "pids": [2, 3, 4],
        "stance_label_3": "support",
        "stance_label_5": "mildly_support",
        "voter_counts": [0, 0, 0, 4, 0],  
        "evidence": [
          10,
          20
        ]
      }
    ]
  }

```