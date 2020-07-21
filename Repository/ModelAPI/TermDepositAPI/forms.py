from django.forms import ModelForm
from . models import Subscriptions

class SubscriptionForm(ModelForm):

	class Meta:
		model = Subscriptions
		fields = '__all__'

	# firstname = forms.CharField(max_length = 15)
	# lastname = forms.CharField(max_length = 15)
	# age = forms.IntegerField(default = 0)
	# job = forms.ChoiceField(max_length = 15,choices = [( 
	# 	('Administrator','Administrator'),
	# 	('Blue Collar','Blue Collar'),
	# 	('Entrepreneur','Entrepreneur'),
	# 	('Housemaid','Housemaid'),
	# 	('Management','Management'),
	# 	('Retired','Retired'),
	# 	('Self Employed','Self Employed'),
	# 	('Services','Services'),
	# 	('Student','Student'),
	# 	('Technician','Technician'),
	# 	('Unemployed','Unemployed'),
	# 	('Unknown','Unknown')
	# )])
	# marital = forms.ChoiceField(max_length = 15,choices = 	[( 
	# 	('Single','Single'),
	# 	('Married','Married'),
	# 	('Divorced','Divorced'),
	# 	('Unknown','Unknown')
	# )])
	# education = forms.ChoiceField(max_length = 15,choices = [(
	# 	('Primary','Primary'),
	# 	('Seconday','Seconday'),
	# 	('Tertiary','Tertiary'),
	# 	('Unknown','Unknown')
	# )])
	# default = forms.ChoiceField(max_length = 15,choices = [(
	# 	('Yes','Yes'),
	# 	('No','No'),
	# )])
	# balance = forms.IntegerField(default = 0)
	# housing = forms.ChoiceField(max_length = 15,choices = [(
	# 	('Yes','Yes'),
	# 	('No','No'),
	# )])
	# loan = forms.ChoiceField(max_length = 15,choices = [(
	# 	('Yes','Yes'),
	# 	('No','No'),
	# )])
	# contact_type = forms.ChoiceField(max_length = 15,choices = [(
	# 	('Cellular','Cellular'),
	# 	('Telephone','Telephone'),
	# 	('Unknown','Unknown'),
	# )])
	# last_contact_day_of_month = forms.IntegerField(default = 0)
	# last_contact_month = forms.ChoiceField(max_length = 15,choices = [(
	# 	('January','January'),
	# 	('February','February'),
	# 	('March','March'),
	# 	('April','April'),
	# 	('May','May'),
	# 	('June','June'),
	# 	('July','July'),
	# 	('August','August'),
	# 	('September','September'),
	# 	('October','October'),
	# 	('November','November'),
	# 	('December','December'),
	# )])
	# last_contact_duration = forms.IntegerField(default = 0)
	# campaign_contacts = forms.IntegerField(default = 0)
	# num_days_passed_after_campaign_comtact = forms.IntegerField(default = 0)
	# previous_contact_before_campaign = forms.IntegerField(default = 0)
	# previous_campaign_outcome = forms.ChoiceField(max_length = 15,choices = [(
	# 	('Success','Success'),
	# 	('Failure','Failure'),
	# 	('Other','Other'),
	# 	('Unknown','Unknown')
	# )])
