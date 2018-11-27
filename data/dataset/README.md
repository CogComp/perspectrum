# Perspective Dataset

## 11.26.2018 Update 
### Task description
Given a claim and a pool of perspectives, find all perspectives that either support or undermine the claim.
### Dataset Format
There are three json files in the folder -- `claim.json`, `perspective.json`, `gold_annotation.json`. 

`claim.json`: A list of claims, each claim have an unique id.
```json
{   
    "id": 2,
    "title": "We ban child performers."
}
```

`perspective.json`: A list of perspective, each perspective have an unique id.
```json
{
    "id": 8134,
    "title": "We Should Have a Quota for Women on Corporate Boards. Quotas are Inherently Helpful"
}
```

`gold_annotation.json`: A list of pairwise gold annotations between an perspective and an claim:
```python
{   
	"claim_id":233,
    "perspective_id":8133,
	"stance": "S"
} 
```

How to read them? refer to the snippet below. 
```python
>>> import json
>>> fin = open("perspective.json", 'r')
>>> perspective_list = json.load(fin)
>>> perspective_list[0]
{"id":1,"title":"Being a performer limits a child\u2019s formal education"}
```
