# Generated by Django 3.0.8 on 2020-07-15 19:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Subscriptions',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('firstname', models.CharField(max_length=15)),
                ('lastname', models.CharField(max_length=15)),
                ('age', models.IntegerField()),
                ('job', models.CharField(choices=[('Administrator', 'Administrator'), ('Blue Collar', 'Blue Collar'), ('Entrepreneur', 'Entrepreneur'), ('Housemaid', 'Housemaid'), ('Management', 'Management'), ('Retired', 'Retired'), ('Self Employed', 'Self Employed'), ('Services', 'Services'), ('Student', 'Student'), ('Technician', 'Technician'), ('Unemployed', 'Unemployed'), ('Unknown', 'Unknown')], max_length=15)),
                ('marital', models.CharField(choices=[('Single', 'Single'), ('Married', 'Married'), ('Divorced', 'Divorced'), ('Unknown', 'Unknown')], max_length=15)),
                ('education', models.CharField(choices=[('Primary', 'Primary'), ('Seconday', 'Seconday'), ('Tertiary', 'Tertiary'), ('Unknown', 'Unknown')], max_length=15)),
                ('default', models.CharField(max_length=15)),
                ('balance', models.IntegerField()),
                ('housing', models.CharField(choices=[('Yes', 'Yes'), ('No', 'No')], max_length=15)),
                ('loan', models.CharField(choices=[('Yes', 'Yes'), ('No', 'No')], max_length=15)),
                ('contact_type', models.CharField(choices=[('Cellular', 'Cellular'), ('Telephone', 'Telephone'), ('Unknown', 'Unknown')], max_length=15)),
                ('last_contact_day_of_the_month', models.IntegerField()),
                ('last_contact_month', models.CharField(choices=[('January', 'January'), ('February', 'February'), ('March', 'March'), ('April', 'April'), ('May', 'May'), ('June', 'June'), ('July', 'July'), ('August', 'August'), ('September', 'September'), ('October', 'October'), ('November', 'November'), ('December', 'December')], max_length=15)),
                ('last_contact_duration_in_seconds', models.IntegerField()),
                ('number_of_contacts_during_campaign', models.IntegerField()),
                ('number_of_days_passed_after_campaign_contact', models.IntegerField()),
                ('previous_contact_before_campaign', models.IntegerField()),
                ('previous_campaign_outcome', models.CharField(choices=[('Success', 'Success'), ('Failure', 'Failure'), ('Other', 'Other'), ('Unknown', 'Unknown')], max_length=15)),
            ],
        ),
    ]
