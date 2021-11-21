#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: optimization.py
'''
Implement structural relaxation using the pretrained E and f models,
and write the optimized structure to CONTCAR file.
'''

import numpy as np
import pickle
import ase.io.vasp
import pymatgen
import os
from matminer.featurizers.site import AGNIFingerprints
from ase.calculators.learning import Learn
from ase.optimize import MDMin
import ase.io

#log file name
log_name='../../data/E_and_f/optimization.log'
#structure file path
path='../../data/E_and_f/optimization/'


def get_dirs(name):
    lines=[]
    for dirs in os.walk(name):
        lines.append(dirs)
    return lines[0][1]


log=open(log_name,'w')
log.write('Structural optimization procedure \n\n\n')
log.close()

#load E models and datas
e_scaler=pickle.load(open('../../data/E_and_f/e_scaler.pkl','rb'))
e_pca=pickle.load(open('../../data/E_and_f/e_pca.pkl','rb'))
e_model=pickle.load(open('../../data/E_and_f/e_model.pkl','rb'))

#load f models and datas
species_type=np.load('../../data/E_and_f/Species_type.npy')
name='agni'
for spe in species_type:
    locals()['f_scaler_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_scaler_%s.pkl' % (name,spe),'rb'))
    locals()['f_pca_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_pca_%s.pkl' % (name,spe),'rb'))
    locals()['f_model_%s_%s' % (name,spe)]=pickle.load(open('../../data/E_and_f/f_%s_model_%s.pkl' % (name,spe),'rb'))
    
log=open(log_name,'a')
log.write('Finish loading models and data \n\n\n')
log.close()

#begin optimization
dirnames=get_dirs(path)
for dirname in dirnames:
    
    log=open(log_name,'a')
    log.write(dirname+'\n\n')
    log.close()    
    
    #gather models
    locals()['f_scaler_%s'% name],locals()['f_pca_%s' % name],locals()['f_model_%s' % name]=[],[],[]
    for spe in species_type:
        locals()['f_scaler_%s'% name].append(locals()['f_scaler_%s_%s' % (name,spe)])
        locals()['f_pca_%s' % name].append(locals()['f_pca_%s_%s' % (name,spe)])
        locals()['f_model_%s' % name].append(locals()['f_model_%s_%s' % (name,spe)])
    
    #initial structure        
    atom_=ase.io.vasp.read_vasp('%s%s/POSCAR' % (path,dirname))
    #set calculator
    atom_.calc = Learn(e_scaler=e_scaler,
                           f_scaler=locals()['f_scaler_%s'% name],
                           e_pca=e_pca,
                           f_pca=locals()['f_pca_%s'% name],
                           e_model=e_model,
                           f_model=locals()['f_model_%s'% name],
                           desc=name,
                           species=species_type)
    
    
    log=open(log_name,'a')
    log.write('Run MDMin ... \n')
    log.close()
    #run structure relaxation
    dyn = MDMin(atom_,trajectory='%s%s/opt.traj' % (path,dirname),logfile=log_name)
    dyn.run(fmax=0.06)
    if dyn.converged():
        log=open(log_name,'a')
        log.write('Structural optimizaiton end \n\n')
        log.close()
    #write the optimized structure to CONTCAR
    ase.io.vasp.write_vasp('%s%s/CONTCAR' % (path,dirname),atom_,vasp5=True,direct=True)




