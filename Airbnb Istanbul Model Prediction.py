import pandas as pd 
import numpy as np
import xlrd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import seaborn as sns
from sklearn.impute import SimpleImputer
from math import exp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error 
import warnings
warnings.filterwarnings('ignore')

class airbnb_istanbul:
	corr_percetage = 0.60

	def __init__(self,df):
		self.df = df
				
	def preprocess(self):
		#They are useless either because it is hard to involve them in the model or have too much missing entries.
		useless = ['id','listing_url','scrape_id', 'last_scraped', 'thumbnail_url','medium_url','picture_url',
           'xl_picture_url','host_id','name','summary','space', 'description','experiences_offered',
           'neighborhood_overview','notes', 'transit','access','interaction','house_rules', 'host_url',
			'host_name','host_location','host_about','host_since', 'host_response_time','host_response_rate','host_acceptance_rate',
			'host_thumbnail_url','host_picture_url','first_review','last_review',
			'host_neighbourhood', 'host_verifications','host_has_profile_pic','host_identity_verified',
			'street', 'market', 'smart_location','country_code','country', 'is_location_exact', 'calendar_last_scraped','city', 'state', 'amenities',
 			'minimum_minimum_nights','maximum_minimum_nights',
 			'minimum_maximum_nights','maximum_maximum_nights',
			'minimum_nights', 'maximum_nights', 
 			'minimum_nights_avg_ntm','maximum_nights_avg_ntm', 'calendar_updated',
 			'number_of_reviews_ltm', 'requires_license','license','jurisdiction_names','neighbourhood_group_cleansed','host_total_listings_count'
 			,'square_feet','weekly_price','monthly_price','security_deposit','zipcode','neighbourhood_cleansed']
		self.df = self.df.drop(useless,axis=1)
		pd.set_option('display.max_columns',None)

		return self.df
	#Finding missing value percentages.	
	def missing_values(self):
		missing_percentage = self.df.isnull().sum()*100/len(self.df)
		plt.figure(figsize = (5,15))
		missing_percentage.plot(kind = 'barh')
		plt.xticks(rotation = 90,fontsize = 10)
		plt.yticks(fontsize = 5)
		plt.xlabel("Missing Percentage",fontsize=14)
		plt.show()

		pd.set_option('display.max_columns',None)

		return missing_percentage

	def to_float(self):
		int_indexes = np.where(self.df.dtypes == 'int64')
		int_labels = self.df.columns[int_indexes[0]]
		self.df[int_labels] = self.df[int_labels].astype('float64')
		self.df['price'] = self.df['price'].apply(lambda s: float(s[1:].replace(',','')))
		self.df['extra_people'] = self.df['extra_people'].apply(lambda s: float(s[1:].replace(',','')))
		#Deleting meaningless entries.
		self.df = self.df[self.df.bedrooms != 0]
		self.df = self.df[self.df.beds != 0]
		self.df = self.df[self.df.bathrooms != 0]
		self.df = self.df[self.df.price != 0]

		return self.df

	def outliers(self):
		number_of_neighbourhoods = self.df.neighbourhood.value_counts()
		counter_number_of_neighbourhoods = Counter(self.df.neighbourhood)
		list_counter_number_of_neighbourhoods = list(Counter(self.df.neighbourhood))
		#Excluding neighbourhoods that places have been rented in,due to being outlier.	
		for i in list_counter_number_of_neighbourhoods:
			if counter_number_of_neighbourhoods[i] < 100:
				del counter_number_of_neighbourhoods[i]
				self.df = self.df[self.df.neighbourhood != i]
		#Assesing outliers in neighbourhood frequency by the help of this graph.
		number_of_neighbourhoods.plot(kind = 'barh')
		plt.title('Number Of Neighbourhood',fontsize = 14)
		plt.xlabel('Frequency',fontsize=14)
		#plt.show()

		return self.df

	def drop_na_row(self):
		self.df = self.df.copy()
		self.df = self.df.dropna(axis = 0)

		return self.df

	def encode_categorical(self):
		non_float_indexes = np.where(self.df.dtypes != 'float64')
		non_float_labels = self.df.columns[non_float_indexes[0]]
		self.df[non_float_labels] = self.df[non_float_labels].apply(LabelEncoder().fit_transform)

		return self.df

	def corr_heat_map(self):
		corr_matrix = np.corrcoef(self.df.T)
		#Finding highly correlated predictors in order to prevent multi collinearity in the data set.
		multi_coll_indexes = np.where(np.logical_and(corr_matrix < 1.0, corr_matrix > self.corr_percetage))
		multi_coll_labels = self.df.columns[multi_coll_indexes[1]]
		ax = sns.heatmap(corr_matrix)
		#plt.show()
		#Dropping variables that are causing multi collinearity.
		self.df = self.df.drop(['calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','bedrooms', 'availability_60',
       'availability_90', 'availability_30',
		'review_scores_accuracy',
		'review_scores_value',
		'require_guest_phone_verification', 'require_guest_profile_picture',
		'host_listings_count'],axis = 1)

		return self.df

	def encoding(self):
		y = self.df['price']
		del self.df['price']
		x = self.df.iloc[:,0:-1]
		#Applying one hot encoder to categorical data.
		non_float_indexes = np.where(x.dtypes != 'float64')
		non_float_labels = x.columns[non_float_indexes[0]]
		self.df = pd.get_dummies(x, prefix = non_float_labels, columns = non_float_labels)
		self.df = self.df.join(y)
		y = np.log(self.df['price'])
		pd.set_option('display.max_columns',None)

		return self.df

	def model(self):
		cv = KFold(n_splits = 5)
		linreg = LinReg()
		y = self.df['price']
		x = self.df.iloc[:,0:-1]
		MAE = []
		R2 = []

		for train, test in cv.split(x):
			linreg.fit(x.iloc[train], y.iloc[train])
			y_predict = linreg.predict(x.iloc[test])
			R2.append(linreg.score(x.iloc[test], y.iloc[test]))
			MAE.append(median_absolute_error(y.iloc[test], y_predict))

		lin_testing_set_score = np.mean(R2) 
		lin_median_abs_error = np.mean(MAE) 

		return R2,MAE

result = airbnb_istanbul(pd.read_csv(r'C:\Users\nezih\Desktop\listings.csv'))

result.preprocess()
#print(result.missing_values())
result.to_float()
result.outliers()
result.drop_na_row()
result.encode_categorical()
result.corr_heat_map()
result.encoding()
print(result.model())
