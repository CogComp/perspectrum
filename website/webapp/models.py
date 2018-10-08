from django.db import models


class Perspective(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()


class Claim(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()


class Evidence(models.Model):
    source = models.CharField(max_length=50)
    content = models.TextField()


class PerspectiveRelation(models.Model):
    """
    Annotation
    """
    GOLD = 'GOLD'
    CLAIM_PERSP_REL = (
        ('S', 'Support'),
        ('U', 'Undermine'),
        ('I', 'Irrelevant'),
        ('N', 'Not Sure')
    )
    
    author = models.CharField(max_length=100)
    claim_id = models.IntegerField()
    perspective_id = models.IntegerField()
    rel = models.CharField(max_length=1, choices=CLAIM_PERSP_REL)
    comment = models.TextField(null=True)


class EvidenceRelation(models.Model):
    """
    Relation annotation between evidence and perspective
    """
    GOLD = 'GOLD'
    author = models.CharField(max_length=100)
    perspective_id = models.IntegerField()
    evidence_id = models.IntegerField()
    comment = models.TextField(null=True)
