#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: 4_E_f_extrapolation_performance.py
'''

Energy and force model
4. Extrapolation performance

'''


#load models
import numpy as np
import pickle
import ase.io.vasp
import pymatgen
import os
from matminer.featurizers.site import AGNIFingerprints
from dscribe.descriptors import SOAP
from sklearn.metrics import mean_absolute_error as mae

log=open('../../data/E_and_f/add_predict.log','w')
log.write('Begin Loading data\n\n')
log.close()

e_scaler=pickle.load(open('../../data/E_and_f/e_scaler.pkl','rb'))
e_pca=pickle.load(open('../../data/E_and_f/e_pca.pkl','rb'))
e_model=pickle.load(open('../../data/E_and_f/e_model.pkl','rb'))
E_MAE,E_MAE_1=[],[]

species_type=np.load('../../data/E_and_f/Species_type.npy')
descriptors=['agni']
for name in descriptors:
    for spe in species_type:
        locals()['f_scaler_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_scaler_%s.pkl' % (name,spe),'rb'))
        locals()['f_pca_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_pca_%s.pkl' % (name,spe),'rb'))
        locals()['f_model_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_model_%s.pkl' % (name,spe),'rb'))
        locals()['F_MAE_%s'% spe],locals()['F_MAE_1_%s'% spe]=[],[]

#get extra structures
path='../../data/E_and_f/add_predict/'

for dirname in [dirs for dirs in os.walk(path)][0][1]:

    log=open('../../data/E_and_f/add_predict.log','a')
    log.write(dirname+'\n')
    log.write('Get new data\n')
    
    #load train set and test set
    e_X_train=np.load('../../data/E_and_f/e_trainx.npy')
    e_y_train=np.load('../../data/E_and_f/e_trainy.npy')
    for name in descriptors:
        for spe in species_type:
            locals()['f_X_train_%s_%s' % (name,spe)]=np.load('./data/f_%s_trainx_%s.npy' % (name,spe))
            locals()['f_y_train_%s_%s' % (name,spe)]=np.load('./data/f_%s_trainy_%s.npy' % (name,spe))
            
    #load extra structures            
    atoms=[]
    i=0
    while True:
        try:
            atoms.append(list(ase.io.vasp.read_vasp_xml('%s%s/vasprun.xml' % (path,dirname),index=i))[0])
            i+=1
        except:
            break
    log.write('Number of new atoms: %d \n' % (len(atoms)))
    
    energies=[struc.get_total_energy()/len(struc.get_atomic_numbers()) for struc in atoms]
    forces=[struc.get_forces() for struc in atoms]
    species=[struc.get_chemical_symbols() for struc in atoms]
    
    #transform atoms to structures for agni
    structures=[]
    for atom in atoms:
        ase.io.vasp.write_vasp('../../data/E_and_f/POSCAR',atom,vasp5=True)
        structures.append(pymatgen.Structure.from_file('../../data/E_and_f/POSCAR'))
    os.remove('../../data/E_and_f/POSCAR')
    log.write('Number of new structures for force model: %d \n\n' % (len(structures)))
    
    #get SOAP for new structure
    soap=SOAP(rcut=8, nmax=5, lmax=5,species=species_type,average='outer')
    n_features=soap.get_number_of_features()
    log.write('Number of new soap features: %s \n' % n_features)
    output=soap.create(system=atoms)
    log.write('Number of new atoms for soap: %d \n\n' % (len(output)))
    
    #get agni for new structure
    etas=[i for i in map(np.log,np.arange(0.8,16,0.2))][2::2][1:]
    n=len(etas)
    agni=AGNIFingerprints(directions=['x','y','z'],etas=etas)
    v=[]
    for i in range(len(structures)):
        x=[]
        #chemical species in a structure
        types_specie=[str(spe) for spe in structures[i].types_of_specie]
        #chemical symbols of sites in a structure
        sub_species=[str(spe) for spe in structures[i].species]
        j=0
        while j < len(structures[i]):
            x1=[]
            for k in range(len(species_type)):
                if species_type[k] in types_specie:
                    structure_new=structures[i].copy()
                    sites=[m for m in range(len(sub_species)) if sub_species[m] != species_type[k] and m != j]
                    h=j-len([l for l in sites if l < j])
                    structure_new.remove_sites(sites)
                    try:
                        x1.append(agni.featurize(structure_new,h))
                    except:
                        x1.append([0]*(3*n))
                else:
                    x1.append([0]*(3*n))    
            x.append(x1)
            j+=1
        v.append(x)
    
    #seperate extra testing set acoording to the chemical element
    log.write('Seperate new agni and force\n')
    force=forces[1:]
    agni=v[1:]
    specie=species[1:]
    for spe in species_type:
        locals()['force_%s' % spe]=[]
        locals()['agni_%s' % spe]=[]
        for i in range(len(force)):
            for j in range(len(force[i])):
                if specie[i][j] == spe:
                    locals()['force_%s' % spe].append(force[i][j])
                    locals()['agni_%s' % spe].append(list(np.array(agni[i][j]).flatten(order='F')))
        log.write(spe+' '+str(np.array(locals()['force_%s' % spe]).shape)+' '+str(np.array(locals()['agni_%s' % spe]).shape)+'\n')
    log.write('\n')
    log.close()
    
    #predict
    log=open('../../data/E_and_f/add_predict.log','a')
    log.write('Predict without adding\n')
    
    #energy
    e_X=e_scaler.transform(output[1:])
    e_X=e_pca.transform(e_X)
    e_predict=e_model.predict(e_X)
    e_mae=mae(energies[1:],e_predict)
    E_MAE.append(e_mae)
    log.write('e_mae: %.3f \n' % e_mae)
    
    #force
    name='agni'
    for spe in species_type:
        locals()['f_X_%s'% spe]=locals()['f_scaler_%s_%s' % (name,spe)].transform(locals()['agni_%s' % spe])
        locals()['f_X_%s'% spe]=locals()['f_pca_%s_%s' % (name,spe)].transform( locals()['f_X_%s'% spe])
        locals()['f_predict_%s'% spe]=locals()['f_model_%s_%s' % (name,spe)].predict( locals()['f_X_%s'% spe])
        locals()['f_mae_%s'% spe]=mae(locals()['force_%s' % spe],locals()['f_predict_%s'% spe])
        locals()['F_MAE_%s'% spe].append(locals()['f_mae_%s'% spe])
        log.write('f_mae %s: %.3f \n' % (spe,locals()['f_mae_%s'% spe]))
    log.write('\n')
    log.close()
    
    
    # Add new data into training set
    #seperate added data acoording to the chemical element
    log=open('../../data/E_and_f/add_predict.log','a')
    log.write('Add one time of new data\n')
    
    add_force=[forces[0]]
    add_agni=[v[0]]
    add_specie=[species[0]]
    for spe in species_type:
        locals()['add_force_%s' % spe]=[]
        locals()['add_agni_%s' % spe]=[]
        for i in range(len(add_force)):
            for j in range(len(add_force[i])):
                if add_specie[i][j] == spe:
                    locals()['add_force_%s' % spe].append(add_force[i][j])
                    locals()['add_agni_%s' % spe].append(list(np.array(add_agni[i][j]).flatten(order='F')))
        log.write(spe+' '+str(np.array(locals()['add_force_%s' % spe]).shape)+' '+str(np.array(locals()['add_agni_%s' % spe]).shape)+'\n')
    log.write('\n')
    
    #add new data to train set
    e_X_train=np.concatenate((e_X_train,np.array([output[0]])))
    e_y_train=np.concatenate((e_y_train,np.array([energies[0]])))
    e_X_train=e_scaler.transform(e_X_train)
    e_X_train=e_pca.transform(e_X_train)
    e_model.fit(e_X_train,e_y_train)
    
    for spe in species_type:
        locals()['f_X_train_%s_%s' % (name,spe)]=np.concatenate((locals()['f_X_train_%s_%s' % (name,spe)],                                                np.array(locals()['add_agni_%s' % spe])))
        locals()['f_y_train_%s_%s' % (name,spe)]=np.concatenate((locals()['f_y_train_%s_%s' % (name,spe)],                                                np.array(locals()['add_force_%s' % spe])))
    
        locals()['f_X_train_%s_%s' % (name,spe)]= locals()['f_scaler_%s_%s' % (name,spe)].transform(locals()['f_X_train_%s_%s' % (name,spe)])
        locals()['f_X_train_%s_%s' % (name,spe)]= locals()['f_pca_%s_%s' % (name,spe)].transform(locals()['f_X_train_%s_%s' % (name,spe)])
        locals()['f_model_%s_%s' % (name,spe)].fit(locals()['f_X_train_%s_%s' % (name,spe)],locals()['f_y_train_%s_%s' % (name,spe)])
    
    #predict
    
    #energy
    e_predict=e_model.predict(e_X)
    e_mae=mae(energies[1:],e_predict)
    E_MAE_1.append(e_mae)
    log.write('e_mae: %.3f \n' % e_mae)
    
    #force
    name='agni'
    for spe in species_type:
        locals()['f_predict_%s'% spe]=locals()['f_model_%s_%s' % (name,spe)].predict( locals()['f_X_%s'% spe])
        locals()['f_mae_%s'% spe]=mae(locals()['force_%s' % spe],locals()['f_predict_%s'% spe])
        locals()['F_MAE_1_%s'% spe].append(locals()['f_mae_%s'% spe])
        log.write('f_mae %s: %.3f \n' % (spe,locals()['f_mae_%s'% spe]))
    log.write('\n')
    log.close()
    
#average MAE    
log=open('../../data/E_and_f/add_predict.log','a')
log.write('mean e_mae: %.3f \n' % np.mean(E_MAE)) 
log.write('mean e_mae_1: %.3f \n' % np.mean(E_MAE_1))
for spe in species_type:
    log.write('mean f_mae %s: %.3f \n' % (spe,np.mean(locals()['F_MAE_%s'% spe])))
    log.write('mean f_mae_1 %s: %.3f \n' % (spe,np.mean(locals()['F_MAE_1_%s'% spe])))

log.write('\n\nFinish\n')
log.close()
