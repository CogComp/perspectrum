# Generated by Django 2.1.1 on 2018-11-27 21:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0016_auto_20181127_2025'),
    ]

    operations = [
        migrations.AddField(
            model_name='claim',
            name='keywords',
            field=models.TextField(default='[]'),
        ),
    ]
