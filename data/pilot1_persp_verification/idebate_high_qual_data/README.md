# Perspective Dataset

## 10.14.2018 Update 
### Task description
Given a claim and a pool of perspectives, find all perspectives that either support or undermine the claim.
### Dataset Format
There are three json files in the folder -- `claims_high_quality.json`, `perspectives_high_quality.json`, `annotations_high_quality.json`. 

`claims_high_quality.json`: A list of claims. Each claim object looks like this:
```json
{   
    "id":1,
    "source":"idebate",
    "title":"The use of child performers should be banned"
}
```

`perspectives_high_quality.json`: A list of perspectives. Each perspective object looks like this:
```json
{
    "id":98,
    "source":"idebate",
    "title":"ACTA is needed to protect brands"
}
```

`annotations_high_quality.json`: A list of annotations between claim-perspective pairs. Each annotation object looks like this:
```python
{   
    "claim_id":2, 
    "perspective_id":14,
    "agreement":1.0,        # Observed agreement for this annotation, can be used as confidence measure
    "annotator_count":3,    # Number of human annotations
    "label":"und"           # There are two possible labels, "sup" (support), "und" (undermine/refute)
} 
```

How to read them? refer to the snippet below. 
```python
> import json
> fin = open("perspectives_high_quality.json", 'r')
> perspective_list = json.load(fin)
> perspective_list[0]
{"id":1,"source":"idebate","title":"Being a performer limits a child\u2019s formal education"}
```
