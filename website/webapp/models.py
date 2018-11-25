from django.db import models


class Perspective(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()
    pilot1_high_agreement = models.BooleanField(default=False)
    similar_persps = models.TextField(default='[]') # Json serialized list of perspective ids


class Claim(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()
    # Human annotation counts for normalizing perspective
    assignment_counts = models.IntegerField(default=0)
    finished_counts = models.IntegerField(default=0)

    # Human annotation counts for evidence verification
    equivalence_assign_counts = models.IntegerField(default=0)
    equivalence_finished_counts = models.IntegerField(default=0)


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
        ('A', 'Slight Support'),
        ('B', 'Slight Undermine'),
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


# User tables
class HITSession(models.Model):
    """
    Record the progress of each HIT session of user
    """
    username = models.CharField(max_length=100)
    instruction_complete = models.BooleanField(default=False)
    job_complete = models.BooleanField(default=False)
    jobs = models.TextField()  # a list of integer claim ids
    finished_jobs = models.TextField()
    duration = models.DurationField()
    last_start_time = models.DateTimeField(null=True) # Used to calculate duration


# Equivalence
class EquivalenceHITSession(models.Model):
    username = models.CharField(max_length=100)
    instruction_complete = models.BooleanField(default=False)
    job_complete = models.BooleanField(default=False)
    jobs = models.TextField()  # a list of integer claim ids
    finished_jobs = models.TextField()
    duration = models.DurationField()
    last_start_time = models.DateTimeField(null=True) # Used to calculate duration


class EquivalenceAnnotation(models.Model):
    session_id = models.IntegerField()
    author = models.CharField(max_length=100)
    perspective_id = models.IntegerField()
    user_choice = models.TextField(default="[]") # A json serialized list of perspective ids


# Experiment results
class ReStep1Results(models.Model):
    claim_id = models.IntegerField()
    perspective_id = models.IntegerField()
    vote_support = models.IntegerField(default=0)
    vote_leaning_support = models.IntegerField(default=0)
    vote_leaning_undermine = models.IntegerField(default=0)
    vote_undermine = models.IntegerField(default=0)
    vote_not_valid = models.IntegerField(default=0)
    p_i_5 = models.FloatField(default=0)
    p_i_3 = models.FloatField(default=0)


# Evidence
class EvidenceHITSession(models.Model):
    username = models.CharField(max_length=100)
    instruction_complete = models.BooleanField(default=False)
    job_complete = models.BooleanField(default=False)
    jobs = models.TextField()  # a list of integer claim ids
    finished_jobs = models.TextField()
    duration = models.DurationField()
    last_start_time = models.DateTimeField(null=True) # Used to calculate duration

