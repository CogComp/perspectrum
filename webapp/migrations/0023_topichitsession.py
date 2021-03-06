# Generated by Django 2.1.1 on 2018-12-07 17:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0022_equivalencebatch'),
    ]

    operations = [
        migrations.CreateModel(
            name='TopicHITSession',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100)),
                ('instruction_complete', models.BooleanField(default=False)),
                ('job_complete', models.BooleanField(default=False)),
                ('jobs', models.TextField()),
                ('finished_jobs', models.TextField()),
                ('duration', models.DurationField()),
                ('last_start_time', models.DateTimeField(null=True)),
            ],
        ),
    ]
