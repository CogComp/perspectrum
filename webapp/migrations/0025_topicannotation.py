# Generated by Django 2.1.1 on 2018-12-07 17:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0024_auto_20181207_1733'),
    ]

    operations = [
        migrations.CreateModel(
            name='TopicAnnotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('author', models.CharField(max_length=100)),
                ('claim_id', models.IntegerField()),
                ('topics', models.TextField(default='[]')),
            ],
        ),
    ]
