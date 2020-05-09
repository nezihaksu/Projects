import pandas as pd
import numpy as np
import xlrd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from math import exp
from sklearn.linear_model import LinearRegression as LinReg
from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_absolute_error, median_absolute_error
import warnings
warnings.filterwarnings('ignore')


class panel_data:

	def __init__(self, df):
		self.df = df

	def stats(self):
		data = self.df
		data1 = data.set_index('yil', 'iller')
		#In Panel regression correlation between variables isn't a vital problem since it merges the longidutinal and time data.Still worth to look at.
		correlation = data1.iloc[:, 0:].corr()
		#Spotting variables that have outliers in different years.
		pd.plotting.scatter_matrix(
			data, alpha=1, diagonal='kde', grid=True, range_padding=0.10)
		plt.show()
		return correlation
	#Finding the number of missing values and their ratio to data set.

	def missing_values(self):
		 missing_data = self.df.iloc[:, :9].isnull().values.tolist()
		 # To flaten the nested list
		 missing_data_flat = list(np.array(missing_data).flat)
		 missing_percent = float(missing_data_flat.count(True)) / \
                     float(len(missing_data_flat))
		 return missing_percent, float(missing_data_flat.count(True))

	#Filling missing values with mean values and panel regress them.
	def preprocessing_regression(self):
		#Filling missing values with mean values.
		imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
		self.df.iloc[:, :9] = imputer.fit_transform(self.df.iloc[:, :9])
		data = self.df.iloc[:, :10]
		#Taking natural log of variable that have outliers
		data.mezun = np.log(self.df.iloc[:, 2])
		data.yogunluk = np.log(self.df.iloc[:, 3])
		data.dogum = np.log(self.df.iloc[:, 4])
		#Setting indexes in order to shape to data into panel form.
		data = data.set_index(['iller', 'yil'])
		#Regressing variables to find out time effect on the relation between regressand and regressors.
		mod = PanelOLS(data.mezun, data.iloc[:, 1:9], time_effects=True)
		res = mod.fit(cov_type='clustered', cluster_entity=True)

		return res


panel_excel = panel_data(pd.read_excel(r"C:\Users\nezih\Desktop\only_turkey_data.xlsx"))
print(panel_excel.stats()
print(panel_excel.missing_values())
print(panel_excel.preprocessing_regression())
