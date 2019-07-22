import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince #FAMD is way to go for dimension reduction rather than PCA and MCA when there the data set is comprised of continuous and categorical variable types.
class live:

	def __init__(self,df):
		self.df = df

	def preprocessing_mca(self):
		data = self.df
		#Number of entries and columns.
		entries = data.shape[0]
		features = data.shape[1]
		#Dropping columns where all the entery values are NaN.
		data.dropna(axis = 1,how = 'all',inplace = True)
		data.drop(['status_id'],axis = 1,inplace = True)
		#Finding out number and percentage of missing values.
		bools = data.isnull().values
		#Turning nested list into one list.
		bools_flaten = list(np.array(bools).flat)
		percentage_empty = float(bools_flaten.count(True))/float(len(bools_flaten))
		#Setting status_publish column into index.	
		data = data.set_index('status_published')
		#FAMD
		famd = prince.FAMD(n_components=2,n_iter=3,copy=True,check_input=True,engine='auto',random_state=42)
		famd = famd.fit(data)
		#Helps to see all columns.  
		pd.set_option('display.max_columns',None)
		ax = famd.plot_row_coordinates(data,ax=None,figsize=(3,3),x_component=0,y_component=1,labels=data.index,
		color_labels =['status_type {}'.format(t) for t in data['status_type']], 
		ellipse_outline=False,
		ellipse_fill=True,
		show_points=True)
		plt.show()
		


		return data.shape[0],data.shape[1],percentage_empty

live_csv = live(pd.read_csv(r"C:\Users\nezih\Desktop\Live.csv"))
print(live_csv.preprocessing_mca())







