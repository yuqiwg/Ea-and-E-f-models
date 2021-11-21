#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: Ea_train_models.py
'''

Ionic migration activation energy model
3. Train models

'''


# 1. Initial data check

# Import data
import pandas as pd
import numpy as np
df_1=pd.read_excel(io='../../data/Ea/M_Ea_1.xlsx',index_col=0)
df_2=pd.read_excel(io='../../data/Ea/M_Ea_2.xlsx',index_col=0)
df_3=pd.read_excel(io='../../data/Ea/M_Ea_3.xlsx',index_col=0)
df_4=pd.read_excel(io='../../data/Ea/M_Ea_4.xlsx',index_col=0)
df0=df_1.append(df_2).append(df_3).append(df_4)

# Replace inf with nan
df0=df0.replace(np.inf,np.nan)

# Drop non-numerical columns
drop=[]
for i in df0.columns:
    if type(df0[i].values[0]) == str:
        drop.append(i)
df0.drop(labels=drop,axis=1,inplace=True)

# Drop data whose BV_Ea is not smaller than 2 eV
index=[i for i in df0.index]
y_init=df0['BV_Ea'].values
drop=[i for i in range(len(y_init)) if y_init[i]>=2]
for i in drop:
    df0.drop(labels=index[i],axis=0,inplace=True,errors='ignore')


# 2. Split test set

from sklearn.model_selection import train_test_split
# Separate the dataset into train and test sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=1)


# 3. Explore data trait with train set

# Import train set
df=train_set

# 3.1 Data cleaning
num_nan=list(df.isnull().sum()) # Number of nan in each column
columns=[]
for i in range(len(num_nan)):
    if num_nan[i] != 0:
        columns.append([df.columns[i],num_nan[i]])

# Lables for drop
drop_0=[] # axis=0
drop_1=[] # axis=1
for i in range(len(columns)):
    if columns[i][1] < 150:
        drop_0.append(columns[i][0])
    else:
        drop_1.append(columns[i][0])

# Drop bad data
df.dropna(subset=drop_0,inplace=True)
df.drop(labels=drop_1,axis=1,inplace=True)

# Labels of features
np.save('../../data/Ea/labels.npy',df.columns)

# Split data to attributes and target
# Attribute
columns=[i for i in df.columns if i != 'BV_Ea']
X_train = df[columns].values
# Target
y_train=df['BV_Ea'].values
np.save('../../data/Ea/Ea_train_x.npy',X_train)
np.save('../../data/Ea/Ea_train_y.npy',y_train)

# 3.2 Standard
from sklearn.preprocessing import StandardScaler
import pickle
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
pickle.dump(scaler, open('../../data/Ea/Ea_scaler.pkl','wb'))


# 4. Choose algorithm

# 4.1 Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_rmse=np.sqrt(mean_squared_error(y_true=y_train, y_pred=lr.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
lr_rmse_scores=np.sqrt(-scores)
print('LR: rmse = %.3f , rmse_cv = %.3f' % (lr_rmse,lr_rmse_scores.mean()))

# 4.2 Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=rf.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(rf, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rf_rmse_scores=np.sqrt(-scores)
print('RF: rmse = %.3f , rmse_cv = %.3f' % (rf_rmse,rf_rmse_scores.mean()))

# 4.3 Multi layer perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=mlp.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(mlp, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
mlp_rmse_scores=np.sqrt(-scores)
print('MLP: rmse = %.3f , rmse_cv = %.3f' % (mlp_rmse,mlp_rmse_scores.mean()))

# 4.4 Support vector machine
from sklearn.svm._classes import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
svr = SVR()
svr.fit(X_train, y_train)
svr_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=svr.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(svr, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
svr_rmse_scores=np.sqrt(-scores)
print('SVR: rmse = %.3f , rmse_cv = %.3f' % (svr_rmse,svr_rmse_scores.mean()))

# 4.5 Kernel ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
krr = KernelRidge()
krr.fit(X_train, y_train)
krr_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=krr.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(krr, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
krr_rmse_scores=np.sqrt(-scores)
print('KRR: rmse = %.3f , rmse_cv = %.3f' % (krr_rmse,krr_rmse_scores.mean()))

# 4.6 KNeighbors
from sklearn.neighbors._regression import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
kn = KNeighborsRegressor()
kn.fit(X_train, y_train)
kn_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=kn.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(kn, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
kn_rmse_scores=np.sqrt(-scores)
print('KN: rmse = %.3f , rmse_cv = %.3f' % (kn_rmse,kn_rmse_scores.mean()))

# 4.7 GradientBoosting
from sklearn.ensemble._gb import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=gbr.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(gbr, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
gbr_rmse_scores=np.sqrt(-scores)
print('GBR: rmse = %.3f , rmse_cv = %.3f' % (gbr_rmse,gbr_rmse_scores.mean()))

# 4.8 DecisionTree
from sklearn.tree._classes import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=dt.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(dt, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
dt_rmse_scores=np.sqrt(-scores)
print('DT: rmse = %.3f , rmse_cv = %.3f' % (dt_rmse,dt_rmse_scores.mean()))

# 4.9 BaggingRegressor
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
br = BaggingRegressor()
br.fit(X_train, y_train)
br_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=br.predict(X_train)))
crossvalidation = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(br, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
br_rmse_scores=np.sqrt(-scores)
print('BR: rmse = %.3f , rmse_cv = %.3f' % (br_rmse,br_rmse_scores.mean()))


# 5. Modify parameters

# 5.1 Get test set
df=X_test
df=df.replace(np.inf,np.nan)
y_test=df['BV_Ea'].values
X_test=df[columns].values

#drop nan data
drop=[]
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if np.isnan(X_test[i][j]):
            drop.append(i)
            break
drop=list(set(drop))
drop.sort()
X_test=np.delete(X_test,drop,axis=0)
y_test=np.delete(y_test,drop,axis=0)
np.save('../../data/Ea/Ea_test_x.npy',X_test)
np.save('../../data/Ea/Ea_test_y.npy',y_test)

X_test=scaler.transform(X_test)

# 5.2 Grid search
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree._classes import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,max_error
from sklearn.model_selection import KFold, cross_val_score

#descriptor names and model names
models_name=['rf','dt']
models=[RandomForestRegressor(n_estimators=200),DecisionTreeRegressor()]
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)

#model parameters
parameters = [dict(random_state=[1,3,5,7],
                   max_depth=[None,2,10,50],
                   max_features=['auto','sqrt'],
                   max_leaf_nodes=[None,3,10],
                   min_samples_split=[2,4,6], 
                   min_samples_leaf=[1,2,3]),
              dict(random_state=[1,3,5,7],
                   max_depth=[None,2,10,50],
                   max_features=['auto','sqrt'],
                   max_leaf_nodes=[None,3,10],
                   min_samples_split=[2,4,6], 
                   min_samples_leaf=[1,2,3])]

#choose best model
for i in range(len(models)):
    model=models[i]
    grid_search = GridSearchCV(model,param_grid=parameters[i],scoring='neg_mean_squared_error',cv=5)
    grid_search.fit(X_train,y_train)
    best_model=grid_search.best_estimator_
    best_score=grid_search.best_score_
    best_params=grid_search.best_params_
    print(models_name[i],best_params,'score: ',best_score)
    pickle.dump(best_model, open('../../data/Ea/Ea_model_%s.pkl' % models_name[i],'wb'))

    #scores of best model
    fit_r2=best_model.score(X_train, y_train)
    fit_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=best_model.predict(X_train)))
    fit_mae=mean_absolute_error(y_true=y_train,  y_pred=best_model.predict(X_train))
    print('Fit results: R2 = %.3f , RMSE = %.3f, MAE = %.3f' % (fit_r2,fit_rmse,fit_mae))

    scores = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
    scores1 = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=1)
    cross_rmse = np.mean([np.sqrt(abs(s)) for s in scores])
    cross_mae = np.mean([abs(s) for s in scores1])
    print('Cross-validation results: Folds: %i, mean RMSE: %.3f, mean MAE: %.3f' % (len(scores),cross_rmse,cross_mae ))

    rmse=np.sqrt(mean_squared_error(best_model.predict(X_test),y_test))
    mae=mean_absolute_error(best_model.predict(X_test),y_test)
    max_e=max_error(best_model.predict(X_test),y_test)
    print('Test results: MAE = %.3f , RMSE = %.3f , MAX ERROR = %.3f' % (mae,rmse,max_e),end='\n') 

