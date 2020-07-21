from rest_framework import serializers
from . models import Subscriptions

class subscriptionsSerializers(serializers.ModelSerializer):
	class Meta(object):
		model = Subscriptions
		fields = '__all__'

