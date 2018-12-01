from django.db import models


class Perspective(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()
    pilot1_high_agreement = models.BooleanField(default=False)
    pilot1_have_stance = models.BooleanField(default=False)
    more_than_two_tokens = models.BooleanField(default=True)
    similar_persps = models.TextField(default='[]') # Json serialized list of perspective ids


class Claim(models.Model):
    source = models.CharField(max_length=50)
    title = models.TextField()
    # Human annotation counts for normalizing perspective
    assignment_counts = models.IntegerField(default=0)
    finished_counts = models.IntegerField(default=0)

    # Human annotation counts for evidence verification
    evidence_assign_counts = models.IntegerField(default=0)
    evidence_finished_counts = models.IntegerField(default=0)

    # Keywords
    keywords = models.TextField(default="[]") # List of string keywords


class Evidence(models.Model):
    source = models.CharField(max_length=50)
    content = models.TextField()
    origin_candidates = models.TextField(default="[]")
    google_candidates = models.TextField(default="[]")


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
    REL = (
        ('S', 'Supported'),
        ('N', 'Not_Supported')
    )
    author = models.CharField(max_length=100)
    perspective_id = models.IntegerField()
    evidence_id = models.IntegerField()
    anno = models.CharField(max_length=1, choices=REL, default='S')
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
    LABEL_5_CHOICES = (
        ('S', 'Support'),
        ('U', 'Undermine'),
        ('A', 'Slight Support'),
        ('B', 'Slight Undermine'),
        ('N', 'Not Valid'),
        ('D', 'Doesnt have Majority Vote')
    )

    LABEL_3_CHOICES = (
        ('S', 'Support'),
        ('U', 'Undermine'),
        ('N', 'Not Valid'),
        ('D', 'Doesnt have Majority Vote')
    )

    claim_id = models.IntegerField()
    perspective_id = models.IntegerField()
    vote_support = models.IntegerField(default=0)
    vote_leaning_support = models.IntegerField(default=0)
    vote_leaning_undermine = models.IntegerField(default=0)
    vote_undermine = models.IntegerField(default=0)
    vote_not_valid = models.IntegerField(default=0)
    label_5 = models.CharField(default='D', max_length=1, choices=LABEL_5_CHOICES)
    label_3 = models.CharField(default='D', max_length=1, choices=LABEL_3_CHOICES)
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


# Evidence Batch for step3
class EvidenceBatch(models.Model):
    evidence_ids = models.TextField(default="[]") # list of integer evidence ids
    assign_counts = models.IntegerField(default=0)
    finished_counts = models.IntegerField(default=0)


# Gold Annotation from step 3
class Step3Results(models.Model):
    LABEL_CHOICES = (
        ('S', 'Support'),
        ('N', 'Not Support'),
        ('D', 'Doesnt have Majority Vote')
    )
    perspective_id = models.IntegerField()
    evidence_id = models.IntegerField()
    vote_support = models.IntegerField(default=0)
    vote_not_support = models.IntegerField(default=0)
    p_i = models.FloatField(default=0)
    label = models.CharField(default='D', max_length=1, choices=LABEL_CHOICES)


# Paraphrase hints for step 2a
class PerspectiveParaphrase(models.Model):
    perspective_id = models.IntegerField()
    user_generated = models.TextField(default="[]")
    session_ids = models.TextField(default="[]")
    hints = models.TextField(default="[]")


# Paraphrase HIT Session
class ParaphraseHITSession(models.Model):
    username = models.CharField(max_length=100)
    instruction_complete = models.BooleanField(default=False)
    job_complete = models.BooleanField(default=False)
    jobs = models.TextField()  # a list of integer claim ids
    finished_jobs = models.TextField()
    duration = models.DurationField()
    last_start_time = models.DateTimeField(null=True) # Used to calculate duration


class ParaphraseBatch(models.Model):
    perspective_ids = models.TextField(default="[]") # list of integer perspective ids
    assign_counts = models.IntegerField(default=0)
    finished_counts = models.IntegerField(default=0)