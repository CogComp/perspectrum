from django.db import models


class Perspective(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()
    evidence = models.TextField()
    

class Claim(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()
    

class RelationAnnotation(models.Model):
    """
    Annotation
    """
    GOLD = 'GOLD'
    CLAIM_PERSP_REL = (
        ('S', 'Support'),
        ('U', 'Undermine'),
        ('I', 'Irrelevant')
    )
    
    author = models.CharField(max_length=100)
    claim_id = models.IntegerField()
    perspective_id = models.IntegerField()
    rel = models.CharField(max_length=1, choices=CLAIM_PERSP_REL)
    comment = models.TextField(null=True)
