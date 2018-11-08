# Perspective Dataset

## 10.14.2018 Update 
### Task description
Given a claim and a pool of perspectives, find all perspectives that either support or undermine the claim.
### Dataset Format
There are three json files in the folder -- `claim_perspective.json`, `perspectives_high_quality.json`, `annotations_high_quality.json`. 

`claim_perspective.json`: A list of perspectves, each perspectve have an unique id.
```json
{   
    "id": 8134,
    "title": "We Should Have a Quota for Women on Corporate Boards. Quotas are Inherently Helpful"
}
```

`evidence.json`: A list of evidences, each evidence have an unique id.
```json
{
    "id":98,
    "content":"Setting quotas will directly lead to more diverse boards and better gender diversity..."
}
```

`gold_annotation.json`: A list of pairwise gold annotations between an perspective and an evidence:
```python
{   
    "perspective_id":8133,
    "evidence_id":8120
} 
```

How to read them? refer to the snippet below. 
```python
>>> import json
>>> fin = open("claim_perspective.json", 'r')
>>> perspective_list = json.load(fin)
>>> perspective_list[0]
{"id":1,"title":"Being a performer limits a child\u2019s formal education"}
```
