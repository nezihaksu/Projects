import pandas as pd
import numpy as np

from pprint import pprint
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


#from textblob import TextBlob

df = pd.read_csv('sentiment_target.csv',engine = 'python')
df.dropna(inplace = True)
df.reset_index(drop = True,inplace = True)

x = df.text
y = df.sentiment 

random_state = 42

x_train,x_test_validation,y_train,y_test_validation = train_test_split(x,y,test_size = 0.02,random_state = random_state)
x_validation,x_test,y_validation,y_test = train_test_split(x_test_validation,y_test_validation,test_size = 0.5,random_state = random_state)

class ClfSwitcher(BaseEstimator):

	def __init__(self, estimator = LogisticRegression()):
	    """
	    A Custom BaseEstimator that can switch between classifiers.
	    :param estimator: sklearn object - The classifier
	    """ 
	    self.estimator = estimator
	
	def fit(self, X, y=None, **kwargs):
	    self.estimator.fit(X, y)
	    return self
	
	def predict(self, X, y=None):
	    return self.estimator.predict(X)
	
	def predict_proba(self, X):
	    return self.estimator.predict_proba(X)
	
	def score(self, X, y):
	    return self.estimator.score(X, y)


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', ClfSwitcher()),
])


feature_range = np.arange(5000,20001,5000)
iter_range = [4000]


log_parameters = {
	'vect__max_df': [0.5, 0.75, 1.0],
    'vect__max_features': feature_range,
    'vect__ngram_range': [(1, 1), (1, 2),(1,3)],  # unigrams or bigrams,trigram
    'tfidf__use_idf': [True, False],
    'tfidf__norm': ['l1', 'l2'], 
    'clf__estimator' : [LogisticRegression()],
    'clf__estimator__penalty': ['l1','l2'],
    'clf__estimator__max_iter': iter_range,
    'clf__estimator__solver':['saga']
   }


sdg_parameter =  {
	'vect__max_df': [0.5, 0.75, 1.0],
    'vect__max_features': feature_range,
    'vect__ngram_range': [(1, 1), (1, 2),(1,3)],  # unigrams or bigrams,trigram
    'tfidf__use_idf': [True, False],
    'tfidf__norm': ['l1', 'l2'], 
	'clf__estimator': [SGDClassifier()], # SVM if hinge loss / logreg if log loss
	'clf__estimator__penalty': ['l2', 'elasticnet', 'l1'],
	'clf__estimator__max_iter': iter_range,
	'clf__estimator__tol': [1e-4],
	'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
}

all_parameters = [log_parameters,sdg_parameter]

best_scores_list = []
best_parameters_list = []

if __name__ == "__main__":

	for parameters in all_parameters:
			
		grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,cv = 5)
		
		print("Performing grid search...")
		print("pipeline:", [name for name, _ in pipeline.steps])
		print("parameters:")
		pprint(parameters)
	    
		t0 = time()
		grid_search.fit(x_train, y_train)
		y_pred = grid_search.predict(x_test)


		print("done in %0.3fs" % (time() - t0))
		print()
		print("Model score with the best model parameters: %0.3f" % grid_search.score(y_pred,y_test))
		best_scores_list.append(grid_search.best_score_)
		print("Best parameters set:")
	    
		best_parameters = grid_search.best_estimator_.get_params()

		best_parameters_list.append(best_parameters)
		
		for param_name in sorted(parameters.keys()):
			print("\t%s: %r" % (param_name, best_parameters[param_name]))


	print(best_scores_list,best_parameters_list)