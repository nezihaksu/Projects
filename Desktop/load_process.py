import pandas as pd
import numpy as np
from numpy import array
import os
from pandas import DataFrame

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix


pos_files_dir = r'C:\Users\nezih\Desktop\txt_sentoken\pos'
neg_files_dir = r'C:\Users\nezih\Desktop\txt_sentoken\neg'


class LoadPreprocess:

	def __init__(self,pos_dir,neg_dir):

		def unicode_converter(text)-> str:
		
			return  "".join([word for word in text if ord(word) < 128])
		
		def load_data():
		
			neg_rev_str = os.listdir(neg_dir)
			pos_rev_str = os.listdir(pos_dir)
			neg_revs,pos_revs = [],[]
			
			for pos_rev in pos_rev_str:
		
				with open(pos_dir + "\\" + str(pos_rev),'r') as positive_file:
		
					pos_revs.append(unicode_converter(positive_file.read()))
		
			for neg_rev in neg_rev_str:
		
				with open(neg_dir + "\\" + str(neg_rev),'r') as negative_file:
		
					neg_revs.append(unicode_converter(negative_file.read()))
		
			return neg_revs,pos_revs
		

		neg_dir,pos_dir = load_data()

		self.pos_text = pos_dir
		self.neg_text = neg_dir



	def vectorizer(self)-> array:
	
		t  = TfidfVectorizer(min_df = 10,max_df = 300,max_features = 4100,stop_words = 'english',token_pattern = r'\w+')
		
		
		vectorized_neg = t.fit_transform(self.neg_text).todense()
		
		vectorized_pos = t.fit_transform(self.pos_text).todense()
		
		num_neg_obs = vectorized_neg.shape[0]
		num_pos_obs = vectorized_pos.shape[0]
		
		
		output_arr = np.concatenate([np.full((num_pos_obs,1),1),np.full((num_neg_obs,1),0)])
		
		features_arr = np.concatenate([vectorized_pos,vectorized_neg]) 
		
		df = pd.DataFrame(data = features_arr)
		
		df['output'] = output_arr
		
		df = df.sample(frac = 1,random_state = 42)
		
		output_df = df['output']
		
		features_df = df.iloc[:,:-1]
	
		return output_df,features_df



def logistic_regression(x:array,y:array,model = LogisticRegression(penalty = 'l1',max_iter = 100)) -> list:

	acc_scores,auc_scores = [],[]

	np.random.seed(0)

	for trial in range(100):
		
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

		model.fit(x_train,y_train)
		
		pred_y_vals = model.predict(x_test)
		
		score = model.score(x_test,y_test)

		acc_scores.append(score)

		pred_prob = model.predict_proba(x_train)

		acc_scores.append(accuracy_score(y_train,pred_y_vals))

		fpr,tpr = roc_curve(y_train,pred_y_vals)[0],roc_curve(y_train,pred_y_vals)[1]
	
		auc_scores.append(auc(fpr,tpr))

		confusion_mat = confusion_matrix(y_train,pred_y_vals)
	
	return acc_scores,auc_scores,pred_prob,confusion_mat


#neg_data,pos_data = load_data()	

target_df,input_df = LoadPreprocess(pos_files_dir,neg_files_dir).vectorizer()

logistic_regression(input_df,target_df)



