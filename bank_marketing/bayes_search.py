import bank_marketing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from lightgbm import LGBMClassifier
from hyperopt import hp,fmin,tpe,Trials,STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import accuracy_score,confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

SEED = 42

df = pd.read_csv('bank_marketing.csv')
df.drop(df.columns.values[0],axis = 1,inplace = True)

categorical_features,df = bank_marketing.convert_categorical(df)
le = LabelEncoder()
#LabelEncoder,because education is an ordinal categorical variable.
#PS: LabelEncoder converts dtype into int32.
df.education =  le.fit_transform(df.education)

x = df.drop('y',axis = 1)
y = df.y

x = pd.get_dummies(x)
y = y.map(dict(yes=1,no=0))

#Balancing the target percentage.
smote = SMOTE(sampling_strategy = 'minority')
x,y = smote.fit_sample(x,y)
sc = MinMaxScaler()
x = sc.fit_transform(x)

def data_splitter(x,y,test = False):
	x_train,x_test_validation,y_train,y_test_validation = train_test_split(x,y,test_size = 0.25,random_state = SEED)
	x_validation,x_test,y_validation,y_test = train_test_split(x_test_validation,y_test_validation,test_size = 0.5,random_state = SEED)

	if test:
		return x_train,y_train,x_validation,y_validation,x_test,y_test

	return x_train,y_train,x_validation,y_validation

x_train,y_train,x_validation,y_validation,x_test,y_test = data_splitter(x,y,test = True)


lgbm = LGBMClassifier

lgbm_params = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}


def hyperopt(param_space,x_train,y_train,x_validation,y_validation,x_test,y_test,num_eval):

	def objective_function(clf_parameters):
		
		clf = lgbm(**clf_parameters)
		score = cross_val_score(clf,x_train,y_train,cv = 5).mean()
	
		return {'loss':-score,'status': STATUS_OK}

	
	trials = Trials()
	best_param = fmin(
		objective_function,
		param_space,
		trials = trials,
		algo = tpe.suggest,
		max_evals = num_eval,
		rstate = np.random.RandomState(SEED))

	loss = [x['result']['loss'] for x in trials.trials]

	best_param_values = [x for x in best_param.values()]

	if best_param_values[0] == 0:
		boosting_type = 'gbdt'

	else:
		boosting_type= 'dart'

	clf_best = 	lgbm(
					learning_rate=best_param_values[2],
					num_leaves=int(best_param_values[5]),
					max_depth=int(best_param_values[3]),
					n_estimators=int(best_param_values[4]),
					boosting_type=boosting_type,
					colsample_bytree=best_param_values[1],
					reg_lambda=best_param_values[6])

	clf_best.fit(x_train,y_train)

	print('Best parameters',best_param)
	print('Test score',clf_best.score(x_validation,y_validation))

	y_pred = clf_best.predict(x_test)

	cm = confusion_matrix(y_test,y_pred)
	ax = plt.subplot()
	sns.heatmap(cm,annot = True,ax = ax)

	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')
	ax.set_title('confusion_matrix')
	ax.xaxis.set_ticklabels(['No','Yes'])
	ax.yaxis.set_ticklabels(['No','Yes'])
	plt.show()

	return trials

num_eval = 40

hyperopt_results = hyperopt(lgbm_params,x_train,y_train,x_validation,y_validation,x_test,y_test,num_eval)

#Test Score:0.944