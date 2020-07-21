from django.db import models

# Create your models here.

class Subscriptions(models.Model):

	JOB_CHOICES = ( 
		('Administrator','Administrator'),
		('Blue Collar','Blue Collar'),
		('Entrepreneur','Entrepreneur'),
		('Housemaid','Housemaid'),
		('Management','Management'),
		('Retired','Retired'),
		('Self Employed','Self Employed'),
		('Services','Services'),
		('Student','Student'),
		('Technician','Technician'),
		('Unemployed','Unemployed'),
		('Unknown','Unknown')
	)

	MARITAL_CHOICES = ( 
		('Single','Single'),
		('Married','Married'),
		('Divorced','Divorced'),
		('Unknown','Unknown')
	)

	EDUCATION_CHOICES = (
		('Primary','Primary'),
		('Seconday','Seconday'),
		('Tertiary','Tertiary'),
		('Unknown','Unknown')
	)

	DEFAULT_CHOICES = (
		('Yes','Yes'),
		('No','No'),
	) 

	LOAN_CHOICES = (
		('Yes','Yes'),
		('No','No'),
	)

	HOUSING_CHOICES = (
		('Yes','Yes'),
		('No','No'),
	)

	CONTACT_TYPE_CHOICES = (
		('Cellular','Cellular'),
		('Telephone','Telephone'),
		('Unknown','Unknown'),
	)

	LAST_CONTACT_MONTH_CHOICES = (
		('January','January'),
		('February','February'),
		('March','March'),
		('April','April'),
		('May','May'),
		('June','June'),
		('July','July'),
		('August','August'),
		('September','September'),
		('October','October'),
		('November','November'),
		('December','December'),
	)

	PREVIOUS_CAMPAIGN_OUTCOME_CHOICES = (
		('Success','Success'),
		('Failure','Failure'),
		('Other','Other'),
		('Unknown','Unknown')
	)

	firstname = models.CharField(max_length = 15)
	lastname = models.CharField(max_length = 15)
	age = models.IntegerField(default = 0)
	job = models.CharField(max_length = 15,choices = JOB_CHOICES)
	marital = models.CharField(max_length = 15,choices = MARITAL_CHOICES)
	education = models.CharField(max_length = 15,choices = EDUCATION_CHOICES)
	default = models.CharField(max_length = 15,choices = DEFAULT_CHOICES)
	balance = models.IntegerField(default = 0)
	housing = models.CharField(max_length = 15,choices = HOUSING_CHOICES)
	loan = models.CharField(max_length = 15,choices = LOAN_CHOICES)
	contact_type = models.CharField(max_length = 15,choices = CONTACT_TYPE_CHOICES)
	last_contact_month = models.CharField(max_length = 15,choices = LAST_CONTACT_MONTH_CHOICES)
	last_contact_day_of_the_month = models.IntegerField(default = 1)
	last_contact_duration_in_seconds = models.IntegerField(default = 0)
	number_of_contacts_during_campaign = models.IntegerField(default = 0)
	number_of_days_passed_after_campaign_contact= models.IntegerField(default = 0)
	previous_contact_before_campaign = models.IntegerField(default = 0)
	previous_campaign_outcome = models.CharField(max_length = 15,choices = PREVIOUS_CAMPAIGN_OUTCOME_CHOICES)

	def __str__(self):
		return '{} {}'.format(self.firstname,self.lastname)