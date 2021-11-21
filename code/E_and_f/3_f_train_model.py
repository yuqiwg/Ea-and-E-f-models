#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: 3_f_train_model.py
'''

Force model
3. Train model

'''

# 1. Data preprocessing

# 1.1 Import all the data
import numpy as np

force=np.load('../../data/E_and_f/Force.npy',allow_pickle=True)
agni=np.load('../../data/E_and_f/F_AGNI.npy',allow_pickle=True)
species=np.load('../../data/E_and_f/F_species.npy',allow_pickle=True)
species_type=np.load('../../data/E_and_f/Species_type.npy',allow_pickle=True)

log=open('../../data/E_and_f/force_train.log','w')
log.write('Numbers of : force %d , agni %d , species %d ,species_type %d \n\n' % (len(force),len(agni),len(species),len(species_type)))
log.close()

# 1.2 Seperate data acoording to the chemical element
log=open('../../data/E_and_f/force_train.log','a')
for spe in species_type:
    locals()['force_%s' % spe]=[]
    locals()['agni_%s' % spe]=[]
    for i in range(len(force)):
        for j in range(len(force[i])):
            if species[i][j] == spe:
                locals()['force_%s' % spe].append(force[i][j])
                locals()['agni_%s' % spe].append(list(np.array(agni[i][j]).flatten(order='F')))
    log.write(spe + ' '+str(np.array(locals()['force_%s' % spe]).shape)+ ' '+str(np.array(locals()['agni_%s' % spe]).shape)+'\n')


# 1.3 Data reproduct

#data pretreatment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

for name in ['agni']:
    for spe in species_type:
# Separate the dataset into training and test sets
        locals()['X_train_%s' % spe], locals()['X_test_%s' % spe], locals()['y_train_%s' % spe], locals()['y_test_%s' % spe] =         train_test_split(locals()['%s_%s' % (name,spe)], locals()['force_%s' % spe], test_size=0.2, random_state=1)
        np.save('./data/f_%s_trainx_%s' % (name,spe),locals()['X_train_%s' % spe])
        np.save('./data/f_%s_testx_%s' % (name,spe),locals()['X_test_%s' % spe])
        np.save('./data/f_%s_trainy_%s' % (name,spe),locals()['y_train_%s' % spe])
        np.save('./data/f_%s_testy_%s' % (name,spe),locals()['y_test_%s' % spe])

#Scale and pca features
        locals()['scaler_%s' % spe] = StandardScaler()
        locals()['X_train_%s' % spe]=locals()['scaler_%s' % spe].fit_transform(locals()['X_train_%s' % spe])
        locals()['pca_%s' % spe] = PCA(0.99999)
        locals()['pca_%s' % spe].fit(locals()['X_train_%s' % spe])
        pickle.dump(locals()['scaler_%s' % spe], open('../../data/E_and_f/f_%s_scaler_%s.pkl' % (name,spe),'wb'))
        pickle.dump(locals()['pca_%s' % spe], open('../../data/E_and_f/f_%s_pca_%s.pkl' % (name,spe),'wb'))
log.write('\nFinish')
log.close()


# 2. Train models

# 2.1 Import all the data
import numpy as np
import pickle

species_type=np.load('./data/Species_type.npy')
for name in ['agni']:
    for spe in species_type:
        locals()['X_train_%s_%s' % (name,spe)]=np.load('../../data/E_and_f/f_%s_trainx_%s.npy' % (name,spe))
        locals()['X_test_%s_%s' % (name,spe)]=np.load('../../data/E_and_f/f_%s_testx_%s.npy' % (name,spe))
        locals()['y_train_%s_%s' % (name,spe)]=np.load('../../data/E_and_f/f_%s_trainy_%s.npy' % (name,spe))
        locals()['y_test_%s_%s' % (name,spe)]=np.load('../../data/E_and_f/f_%s_testy_%s.npy' % (name,spe))
        locals()['scaler_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_scaler_%s.pkl' % (name,spe),'rb'))
        locals()['pca_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_pca_%s.pkl' % (name,spe),'rb'))

# 2.2 Scaler and pca
for name in ['agni']:
    for spe in species_type:
        locals()['X_train_%s_%s' % (name,spe)]=locals()['scaler_%s_%s' % (name,spe)].transform(locals()['X_train_%s_%s' % (name,spe)])
        locals()['X_test_%s_%s' % (name,spe)]=locals()['scaler_%s_%s' % (name,spe)].transform(locals()['X_test_%s_%s' % (name,spe)])
        locals()['X_train_%s_%s' % (name,spe)]=locals()['pca_%s_%s' % (name,spe)].transform(locals()['X_train_%s_%s' % (name,spe)])
        locals()['X_test_%s_%s' % (name,spe)]=locals()['pca_%s_%s' % (name,spe)].transform(locals()['X_test_%s_%s' % (name,spe)])

# 2.3 Grid search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error,max_error
from sklearn.model_selection import KFold, cross_val_score

#descriptor names and model names
descriptors=['agni']
models=[KernelRidge()]
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)

#model parameters
flo=[1e-0,1e-1,1e-2,1e-3,1e-4,1e-5]
parameters = [dict(kernel=['rbf'],gamma=flo,alpha=flo)]

log=open('../../data/E_and_f/force_model.log','w')
log.write("Begin grid search \n")
log.close()

for name in descriptors:
    for spe in species_type:
        X_train=locals()['X_train_%s_%s' % (name,spe)]
        X_test=locals()['X_test_%s_%s' % (name,spe)]
        y_train=locals()['y_train_%s_%s' % (name,spe)]
        y_test=locals()['y_test_%s_%s' % (name,spe)]
        
#choose best model
        for i in range(len(models)):
            model=models[i]
            grid_search = GridSearchCV(model,param_grid=parameters[i],scoring='neg_mean_squared_error',cv=5)
            grid_search.fit(X_train,y_train)
            final_model=grid_search.best_estimator_
            final_score=grid_search.best_score_
            final_params=grid_search.best_params_
        log=open('../../data/E_and_f/force_model.log','a')
        log.write('Force %s %s score: %.3f \n' % (spe,str(final_params),final_score))
        log.close()
        pickle.dump(final_model, open('../../data/E_and_f/f_%s_model_%s.pkl' % (name,spe),'wb'))
        
#scores of best model
        fit_r2=final_model.score(X_train, y_train)
        fit_rmse=np.sqrt(mean_squared_error(y_true=y_train,  y_pred=final_model.predict(X_train)))
        fit_mae=mean_absolute_error(y_true=y_train,  y_pred=final_model.predict(X_train))
        log=open('../../data/E_and_f/force_model.log','a')
        log.write('Fit results: R2 = %.3f , RMSE = %.3f, MAE = %.3f \n' % (fit_r2,fit_rmse,fit_mae))
        log.close()

        scores = cross_val_score(final_model, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
        scores1 = cross_val_score(final_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=1)
        cross_rmse = np.mean([np.sqrt(abs(s)) for s in scores])
        cross_mae = np.mean([abs(s) for s in scores1])
        log=open('../../data/E_and_f/force_model.log','a')
        log.write('Cross-validation results: Folds: %i, mean RMSE: %.3f, mean MAE: %.3f \n' % (len(scores),cross_rmse,cross_mae ))
        log.close()

        rmse=np.sqrt(mean_squared_error(final_model.predict(X_test),y_test))
        mae=mean_absolute_error(final_model.predict(X_test),y_test)
        log=open('../../data/E_and_f/force_model.log','a')
        log.write('Test results: MAE = %.3f , RMSE = %.3f \n\n' % (mae,rmse)) 
        log.close()
