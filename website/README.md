# Webapp 

## Getting started 

To install the python dependencies, simply do: 
```python 
pip3.6 install -r requirements.txt
```

- Check if Django is installed:
 ```
 $ python -m django --version
 ```
 
 - Run the app: 
```
$ python3.7 manage.py runserver
```

## Immigration of Models to DB 
If starting with fresh DB, you need to create tables corresponding to the models in the DB:  
```
> python3.6 manage.py makemigrations app  # you should see the migrations under "app/migrations"
> python3.6 manage.py sqlmigrate app 0001
> python3.6 manage.py migrate 
```

**Note** if you change the models (add or drop a table), to synch up the models with the DB, run this: 
```
python3.6 manage.py migrate --run-syncdb
```