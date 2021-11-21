#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: 3_E_train_model.py
'''

Energy model
3. Train model

'''


#import all the data
import numpy as np

energy=np.load('../../data/E_and_f/Energy.npy',allow_pickle=True)
soap=np.load('../../data/E_and_f/E_SOAP.npy',allow_pickle=True)
log=open('../../data/E_and_f/energy_train_model.log','w')
log.write('Shape of energies: %s , Shape of soap: %s \n\n' % (str(np.array(energy).shape),str(np.array(soap).shape)))
log.close()

#split, scaler and pca
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(soap, energy, test_size=0.2, random_state=1)
np.save('../../data/E_and_f/e_trainx',X_train)
np.save('../../data/E_and_f/e_testx',X_test)
np.save('../../data/E_and_f/e_trainy',y_train)
np.save('../../data/E_and_f/e_testy',y_test)

#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform (X_test)

#pca
pca = PCA(0.99999)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

pickle.dump(scaler, open('../../data/E_and_f/e_scaler.pkl','wb'))
pickle.dump(pca, open('../../data/E_and_f/e_pca.pkl','wb'))

#initial grid search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, random_state=1)
parameters = dict(max_depth=[None,2,4,10],
                   max_features=['auto','sqrt','log2'],
                   min_samples_split=[2,6,10], 
                   min_samples_leaf=[1,3,5])

log=open('../../data/E_and_f/energy_train_model.log','a')
log.write('Begin grid search\n\n')
log.close()

grid_search = GridSearchCV(rf,param_grid=parameters,scoring='neg_mean_squared_error',cv=5)
grid_search.fit(X_train,y_train)

log=open('../../data/E_and_f/energy_train_model.log','a')

log.write('Energy soap:\n')
log.write(str(grid_search.best_params_)+'  score:  '+str(grid_search.best_score_)+'\n\n')
best_model=grid_search.best_estimator_
pickle.dump(best_model, open('../../data/E_and_f/e_model.pkl','wb'))
log.close()

#model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error,max_error
from sklearn.model_selection import KFold, cross_val_score

log=open('../../data/E_and_f/energy_train_model.log','a')
fit_r2=best_model.score(X_train, y_train)
fit_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=best_model.predict(X_train)))
fit_mae=mean_absolute_error(y_true=y_train,  y_pred=best_model.predict(X_train))
log.write('Fit results: R2 = %.3f , RMSE = %.3f, MAE = %.3f \n' % (fit_r2,fit_rmse,fit_mae))

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
scores1 = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=1)
cross_rmse = np.mean([np.sqrt(abs(s)) for s in scores])
cross_mae = np.mean([abs(s) for s in scores1])
log.write('Cross-validation results: Folds: %i, mean RMSE: %.3f, mean MAE: %.3f \n' % (len(scores),cross_rmse,cross_mae ))

rmse=np.sqrt(mean_squared_error(best_model.predict(X_test),y_test))
mae=mean_absolute_error(best_model.predict(X_test),y_test)
max_e=max_error(best_model.predict(X_test),y_test)
log.write('Test results: MAE = %.3f , RMSE = %.3f , MAX ERROR = %.3f \n\n' % (mae,rmse,max_e))
log.write('Finish\n')
log.close()
