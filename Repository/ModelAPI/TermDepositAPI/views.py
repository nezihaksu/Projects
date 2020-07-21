from django.shortcuts import render
from . forms import SubscriptionForm
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.contrib import messages
from .forms import SubscriptionForm
from django.http import HttpResponse
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from . models import Subscriptions
from .serializers import subscriptionsSerializers
import joblib
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd

# Create your views here.

class SubscriptionView(viewsets.ModelViewSet):
	queryset = Subscriptions.objects.all()
	serializer_class = subscriptionsSerializers

def convert_ohe(df):
	one_hot_col = joblib.load(r'C:\Users\nezih\Repository\ModelAPI\TermDepositAPI\one_hot.pkl')
	categorical_names = ['job','marital','education','default','housing','loan','last_contact_month','contact_type','previous_campaign_outcome']
	df_preprocessed = pd.get_dummies(df,columns = categorical_names)
	print('convert_ohe columns',df_preprocessed.columns)
	new_dict = {}
	for col in one_hot_col:
		if col in df_preprocessed.columns:
			new_dict[col] = df_preprocessed[col].values
		else:
			new_dict[col] = 0
	new_df = pd.DataFrame(new_dict)
	print('newdf shape')
	print(new_df.shape)
	return new_df


#@api_view(["POST"])
def subscription_status(unit):
	try:
		model = joblib.load(r'C:\Users\nezih\Repository\ModelAPI\TermDepositAPI\term_deposit_model.pkl')
		scalers = joblib.load(r'C:\Users\nezih\Repository\ModelAPI\TermDepositAPI\term_deposit_data_scaler.pkl')
		x = scalers.transform(unit)
		y_pred = model.predict(x)
		y_prob = model.predict_proba(x)
		newdf = pd.DataFrame(y_pred,columns = ['Subscription'])
		newdf = newdf.replace({1:'May Subscribe',0:'May Not Subscribe'})
		return ('Client {}'.format(newdf))
	except ValueError as e:
		return Response(e.args[0])


def cxcontact(request):
	if request.method == 'POST':
		form = SubscriptionForm(request.POST)
		if form.is_valid():
			firstname = form.cleaned_data['firstname']
			lastname = form.cleaned_data['lastname']
			age = form.cleaned_data['age']
			job = form.cleaned_data['job']
			marital = form.cleaned_data['marital']
			education = form.cleaned_data['education']
			default = form.cleaned_data['default']
			balance = form.cleaned_data['balance']
			housing = form.cleaned_data['housing']
			loan = form.cleaned_data['loan']
			contact = form.cleaned_data['contact_type']
			month = form.cleaned_data['last_contact_month']
			day = form.cleaned_data['last_contact_day_of_the_month']
			duration = form.cleaned_data['last_contact_duration_in_seconds']
			campaign = form.cleaned_data['number_of_contacts_during_campaign']
			pdays = form.cleaned_data['number_of_days_passed_after_campaign_contact']
			previous = form.cleaned_data['previous_contact_before_campaign']
			poutcome = form.cleaned_data['previous_campaign_outcome']
			data_dict = (request.POST).dict()
			df = pd.DataFrame(data_dict,index = [0])
			print(df.columns)
			answer = subscription_status(convert_ohe(df))
			print(answer)
			messages.success(request,'Subscription status: {}'.format(answer))
			

	form = SubscriptionForm()

	return render(request,'subscriptionform/cxform.html',{'form':form})