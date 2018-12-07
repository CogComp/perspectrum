## Details of Lucene experiment 

- elasticsearch version used: 6.3.2. After downloading this version, run it from commandline: 
```
./elasticsearch
```

If you want to make the server available over the network, make the proper changes in the config file: 
```
network.host: 0.0.0.0
http.port: 8080
discovery.type: single-node
```

Then create an index `perspectivesAndClaims``: 
```
curl -X PUT "localhost:9200/perspectivesandclaims"
```

which should make it available on the following uri: 
```
http://localhost:9200
```

To make sure the index is there, list all the indices: 
```
http://bronte.cs.illinois.edu:8080/_cat/indices
```

Install `elasticsearch_loader` to move the questions to the elasticsearch: 
```
pip install elasticsearch_loader
```

And copy the files: 
```
elasticsearch_loader --index perspectivesandclaims --es-host http://bronte.cs.illinois.edu:8080  --type text json /shared/shelley/khashab2/perspective/data/pilot3_twowingos/110718-training-data/claim_perspective.json
```

if anything happens middle of the experiment and you want to delete the existing index, you can use commandline: 
```
curl -XDELETE 'http://localhost:9200/perspectivesAndClaims'
```
