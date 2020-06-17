import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('sentiment_target.csv',engine = 'python')
df.dropna(inplace = True)
df.reset_index(drop = True,inplace = True)

x = df.text
y = df.sentiment 

random_state = 42

x_train,x_test_validation,y_train,y_test_validation = train_test_split(x,y,test_size = 0.02,random_state = random_state)
x_validation,x_test,y_validation,y_test = train_test_split(x_test_validation,y_test_validation,test_size = 0.5,random_state = random_state)

vectorizer = TfidfVectorizer(max_features = 100000,ngram_range = (1,3))
x_train_vect = vectorizer.fit_transform(x_train)
x_validation_vect = vectorizer.transform(x_validation)

chi2_results = {}

for n in np.arange(10000,100001,10000):

	kbest = SelectKBest(chi2,k=n)
	x_train_kbest_selected = kbest.fit_transform(x_train_vect,y_train)
	x_validation_kbest_selected = kbest.transform(x_validation_vect)
	clf = LogisticRegression(penalty = 'l2',solver = 'saga',max_iter = 1000)
	clf.fit(x_train_kbest_selected,y_train)
	score = clf.score(x_validation_kbest_selected,y_validation)
	
	chi2_results.update({n:score})

	print('Chi2 feature selection calculated for {} features.'.format(n))

print(chi2_results)

#For 80000 features model yields the maximum score,which is 82.55.
#In the 80000 features there are the most useful features to predict labels.