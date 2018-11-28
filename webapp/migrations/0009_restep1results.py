# Generated by Django 2.1.1 on 2018-11-21 04:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0008_auto_20181117_2055'),
    ]

    operations = [
        migrations.CreateModel(
            name='ReStep1Results',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('claim_id', models.IntegerField()),
                ('perspective_id', models.IntegerField()),
                ('vote_support', models.IntegerField(default=0)),
                ('vote_leaning_support', models.IntegerField(default=0)),
                ('vote_leaning_undermine', models.IntegerField(default=0)),
                ('vote_undermine', models.IntegerField(default=0)),
                ('vote_not_valid', models.IntegerField(default=0)),
                ('p_i_5', models.FloatField(default=0)),
                ('p_i_3', models.FloatField(default=0)),
            ],
        ),
    ]