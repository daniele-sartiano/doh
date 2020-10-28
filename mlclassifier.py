#!/usr/bin/env python

import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, roc_auc_score, precision_recall_fscore_support, roc_curve, auc, f1_score, recall_score


from reader import Reader, SEED

class Estimator:
    def __init__(self, reader, filename):
        self.reader = reader
        self.filename = filename

    def fit_params(self, data):
        return {}

    def save(self, model):
        data = {'features': self.reader.features}
        with open('data/data.pickle', 'wb') as fout:
            pickle.dump(data, fout)
        data = {'model': model}
        with open(self.filename, 'wb') as fout:
            pickle.dump(data, fout)

    @staticmethod
    def get_data():
        with open('data/data.pickle', 'rb') as fin:
            return pickle.load(fin)


class BRandomForestClassifier(Estimator):

    def test(self, clf, data):
        y_pred = clf.predict(data['val']['X'])
        print(y_pred)
        #y_pred_proba = np.array(clf.predict_proba(val))

        print(y_pred)
        best_preds = np.asarray([np.argmax(line) for line in y_pred])

        #print(best_preds)
        
        print(classification_report(data['val']['y'], best_preds))
        print(confusion_matrix(data['val']['y'], best_preds))

        auc = roc_auc_score(data['val']['y'], y_pred, multi_class='ovr')
        print('auc', auc)
        
    
    def train(self):    
        data = self.reader.read()
    
        clf = BalancedRandomForestClassifier(max_depth=2, random_state=SEED)
        clf.fit(data['train']['X'], data['train']['y'])

        self.test(clf, data)
    

class XGBoostClassifier(Estimator):

    def test(self, clf, data):        
        val = xgb.DMatrix(data['val']['X'].values, data['val']['y'].values)
        y_pred = clf.predict(val)
        print('y_pred', y_pred)
        #y_pred_proba = np.array(clf.predict_proba(val))
        #best_preds = np.asarray([np.argmax(line) for line in y_pred])
        best_preds = [round(value) for value in y_pred]

        print(classification_report(val.get_label(), best_preds))
        print(confusion_matrix(val.get_label(), best_preds))

        auc = roc_auc_score(val.get_label(), y_pred, multi_class='ovr')
        print('auc', auc)    

    def train(self):
        data = self.reader.read()
        params = {
            'learning_rate': 0.05,
            'objective':'binary:logistic',
            'min_child_weight': 1,
            'max_depth': 3,
            'colsample_bytree': 0.3,
            'booster': 'gbtree',
            'seed': SEED,
            'eval_metric': 'auc',
            'n_jobs': 30
        }

        params['gpu_id'] = 2
        params['tree_method'] = 'gpu_hist'

        negative = data['train']['y'].value_counts()[0]
        positive = data['train']['y'].value_counts()[1]
        params['scale_pos_weight'] = negative/positive

        #params['scale_pos_weight'] = 1
        train = xgb.DMatrix(data['train']['X'].values, data['train']['y'].values)
        val = xgb.DMatrix(data['val']['X'].values, data['val']['y'].values)
        
        #watchlist = [(train, 'train'), (val, 'eval')]
        watchlist = [(val, 'eval')]
        
        epochs = 10000
        
        clf = xgb.train(params, train, epochs, evals=watchlist, early_stopping_rounds=1000, verbose_eval=True)

        auc = self.test(clf, data)        
        self.save(clf)

    def grid(self):
        data = self.reader.read()

        model = xgb.XGBClassifier()

        weights = [1, 10, 25, 50, 75, 99, 100, 1000]
        param_grid = dict(scale_pos_weight=weights)
        param_grid['gpu_id'] = [0]
        param_grid['tree_method'] = ['gpu_hist']

        
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

        # define grid search
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=10, cv=cv, scoring='roc_auc')
        # execute the grid search
        grid_result = grid.fit(data['train']['X'], data['train']['y'])

        # report the best configuration
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # report all configurations
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    

def main():
    reader = Reader('data/extracted-features')
    x = XGBoostClassifier(reader, 'test.pickle')
    #x = BRandomForestClassifier(reader, 'test.pickle')
    x.train()
    #x.grid()
    

if __name__ == '__main__':
    main()
