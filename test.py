from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
import os 
from sklearn import datasets
from sklearn.cross_validation import *
from sklearn import metrics
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
output_path = './output/'

def hyperopt_train_test(params, trial):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = RandomForestClassifier(**params)
    train_pred = cross_val_predict(clf,X,y,cv=3)
    df_train_pred = pd.DataFrame({'train_pred': train_pred})
    df_train_pred.to_csv(os.path.join(output_path, 'train_pred_trial_%d.csv' % trial), encoding="utf8")
    
    return metrics.accuracy_score(y, train_pred) 

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

space4rf1 = {
    'max_depth': 20, 
    'max_features': 2,
    'n_estimators': 20, 
     #'criterion': hp.choice('criterion', ["gini", "entropy"]),
     #'scale': hp.choice('scale', [0, 1]),
     #'normalize': hp.choice('normalize', [0, 1])
}

best = 0
count = 0
def f(params):
    global best
    global count
    count += 1
    acc = hyperopt_train_test(params, count)
    if acc > best:
        best = acc
        print('new best %s, %s' % (str(best), str(params)))
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf1, algo=tpe.suggest, max_evals=300, trials=trials)
print("best %s" %  str(best))
print(count)
