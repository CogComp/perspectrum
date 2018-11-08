## Details of Lucene experiment 

- elasticsearch version used: 6.3.2. After downloading this version, run it from commandline: 
```
./elasticsearch
```

Then create an index `perspectivesAndClaims``: 
```
curl -X PUT "localhost:9200/perspectivesAndClaims"
```

which should make it available on the following uri: 
```
http://localhost:9200
```

Install `elasticsearch_loader` to move the questions to the elasticsearch: 
```
pip install elasticsearch_loader
```

And copy the files: 
```
elasticsearch_loader --index perspectivesAndClaims --type text json claim_perspective.json
```

With questions indexed, you're good to run the system on new questions. 



if anything happens middle of the experiment and you want to delete the existing index, you can use commandline: 
```
curl -XDELETE 'http://localhost:9200/perspectivesAndClaims'
```

