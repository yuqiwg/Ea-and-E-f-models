#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: Ea_E_predict_LiMgBiOS.py
'''

Ionic migration activation energy model
Predict Ea and energy of Li(1-2x)Mg(x)BiOS

'''


# 1. Data retrieve

# 1.1 Load the Ea dataset

import os
def get_dirs(name):
    lines=[]
    for dirs in os.walk(name):
        lines.append(dirs)
    return lines[0][1]
path='../../data/E_and_f/optimization/'
dirnames=get_dirs(path)

# 1.2 Add basic properties
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df=pd.DataFrame()
icsd=dirnames
ion=len(icsd)*['Li']
mot_val=[]
mot_rad=[]
mot_ele=[]
for i in ion:
    if i in ['Ag', 'Li','Na','K']:
        mot_val.append(1)
        if i == 'Ag':
            mot_rad.append(1.15)
            mot_ele.append(5.23)
        elif i == 'Li':
            mot_rad.append(0.76)
            mot_ele.append(14.59)
        elif i == 'Na':
            mot_rad.append(1.02)
            mot_ele.append(9.44)
        else:
            mot_rad.append(1.38)
            mot_ele.append(6.47)
    elif i == 'Al':
        mot_val.append(3)
        mot_rad.append(0.535)
        mot_ele.append(26.72)
    else:
        mot_val.append(2)
        if i == 'Ca':
            mot_rad.append(1)
            mot_ele.append(11.3)
        elif i == 'Mg':
            mot_rad.append(0.72)
            mot_ele.append(17.13)
        else:
            mot_rad.append(0.74)
            mot_ele.append(4.92)
        df['mot_atom']=pd.Series(ion,index=icsd)
df['mot_val']=pd.Series(mot_val,dtype=float,index=icsd)
df['mot_rad']=pd.Series(mot_rad,dtype=float,index=icsd)
df['mot_ele']=pd.Series(mot_ele,dtype=float,index=icsd)

#add structure
import pymatgen as py
structure=[]
drop1=[]
for i in range(len(df.index)):
    try:
        structure.append(py.Structure.from_file('%s%s/CONTCAR' % (path,df.index[i])))
    except:
        drop1.append(i)
for j in [df.index[i] for i in drop1]:
    try:
        df.drop(labels=j,axis=0,inplace=True)
    except:
        continue
df['structure']=pd.Series(structure,index=df.index)

#add composition
df['formula']=pd.Series([stru.formula for stru in df['structure']],index=df.index)
df['composition']=pd.Series([stru.composition for stru in df['structure']],index=df.index)


# 2. Featurize

# 2.1 Descriptors for whole

# OxidCompsition
from matminer.featurizers.conversions import CompositionToOxidComposition as CTOC
oc=CTOC(target_col_id='composition_oxid',overwrite_data=False,coerce_mixed=True,return_original_on_error=False)
df=oc.featurize_dataframe(df,'composition',ignore_errors=True)

# AtomicObitals
from matminer.featurizers.composition import AtomicOrbitals as AO
ao=AO()
df=ao.featurize_dataframe(df,'composition',ignore_errors=True)

# BandCenter
from matminer.featurizers.composition import BandCenter as BC
bc=BC()
df=bc.featurize_dataframe(df,'composition',ignore_errors=True)

# OxidationStates
from matminer.featurizers.composition import OxidationStates as OS
os=OS()
df=os.featurize_dataframe(df,'composition_oxid',ignore_errors=True)
df=df[df['minimum oxidation state']<0] 

# ElectronegativityDiff
from matminer.featurizers.composition import ElectronegativityDiff as ED
ed=ED()
df=ed.featurize_dataframe(df,'composition_oxid',ignore_errors=True)

# Stoichiometry
from matminer.featurizers.composition import Stoichiometry
stochi=Stoichiometry(p_list=(0, 2, 3, 5, 7, 10), num_atoms=False)
df=stochi.featurize_dataframe(df,'composition',ignore_errors=True)

# YangSolidSolution
from matminer.featurizers.composition import YangSolidSolution as YSS
yss=YSS()
df=yss.featurize_dataframe(df,'composition',ignore_errors=True)

# ElementFraction
from matminer.featurizers.composition import ElementFraction as EF
ef=EF()
df=ef.featurize_dataframe(df,'composition_oxid',ignore_errors=True)

# IonProperty
from matminer.featurizers.composition import IonProperty
from matminer.utils.data import PymatgenData
data_source=PymatgenData()
ionpro=IonProperty(data_source, fast=False)
df=ionpro.featurize_dataframe(df,'composition_oxid',ignore_errors=True)

# ValenceOrbital
from matminer.featurizers.composition import ValenceOrbital
vo=ValenceOrbital(orbitals=('s', 'p', 'd', 'f'), props=('avg', 'frac'))
df=vo.featurize_dataframe(df,'composition',ignore_errors=True)

# BondFractions
from matminer.featurizers.structure import BondFractions
from pymatgen.analysis.local_env import CrystalNN
bondtypes=['Ag - H','Ag - N','Ag - O','Ag - S','Ag - Se','Ag - Te','Ag - F','Ag - Cl','Ag - Br','Ag - I',
           'Al - H','Al - N','Al - O','Al - S','Al - Se','Al - Te','Al - F','Al - Cl','Al - Br','Al - I',
           'Ca - H','Ca - N','Ca - O','Ca - S','Ca - Se','Ca - Te','Ca - F','Ca - Cl','Ca - Br','Ca - I',
           'K - H','K - N','K - O','K - S','K - Se','K - Te','K - F','K - Cl','K - Br','K - I',
           'Li - H','Li - N','Li - O','Li - S','Li - Se','Li - Te','Li - F','Li - Cl','Li - Br','Li - I',
           'Mg - H','Mg - N','Mg - O','Mg - S','Mg - Se','Mg - Te','Mg - F','Mg - Cl','Mg - Br','Mg - I',
           'Na - H','Na - N','Na - O','Na - S','Na - Se','Na - Te','Na - F','Na - Cl','Na - Br','Na - I',
            'Zn - H','Zn - N','Zn - O','Zn - S','Zn - Se','Zn - Te','Zn - F','Zn - Cl','Zn - Br','Zn - I']
bondf=BondFractions(nn=CrystalNN(), bbv=0, no_oxi=False, approx_bonds=False, token=' - ', allowed_bonds=bondtypes)
df=bondf.featurize_dataframe(df,'structure',ignore_errors=True)

# GlobalSymmetryFeatures
from matminer.featurizers.structure import GlobalSymmetryFeatures
gsf=GlobalSymmetryFeatures()
df=gsf.featurize_dataframe(df,'structure',ignore_errors=True)

# 2.2 Descriptors for migrator

# AverageBondAngle
from matminer.featurizers.site import AverageBondAngle
from pymatgen.analysis.local_env import VoronoiNN
nn=VoronoiNN()
avgb=AverageBondAngle(nn)
labels=avgb.feature_labels()
avgb_mean=[]
for i in range(len(df.structure)):
    x=[]
    for j in range(len(df.structure[i])):
        if df.structure[i][j].as_dict()['species'][0]['element'] == df.mot_atom[i]:
            x.append(avgb.featurize(df.structure[i],j))
    avgb_mean.append(np.mean(x,0))
for i in range(0,len(labels)):
    avgb_fea=[x[i] for x in avgb_mean]
    df['MIG_'+labels[i]]=avgb_fea

# AverageBondLength
from matminer.featurizers.site import AverageBondLength
from pymatgen.analysis.local_env import VoronoiNN
nn=VoronoiNN()
avgl=AverageBondLength(nn)
labels=avgl.feature_labels()
avgl_mean=[]
for i in range(len(df.structure)):
    x=[]
    for j in range(len(df.structure[i])):
        if df.structure[i][j].as_dict()['species'][0]['element'] == df.mot_atom[i]:
            x.append(avgl.featurize(df.structure[i],j))
    avgl_mean.append(np.mean(x,0))
for i in range(0,len(labels)):
    avgl_fea=[x[i] for x in avgl_mean]
    df['MIG_'+labels[i]]=avgl_fea

# IntersticeDistribution
from matminer.featurizers.site import IntersticeDistribution
indis=IntersticeDistribution(interstice_types=['dist','vol'])
labels=indis.feature_labels()
indis_mean=[]
for i in range(len(df.structure)):
    x=[]
    for j in range(len(df.structure[i])):
        if df.structure[i][j].as_dict()['species'][0]['element'] == df.mot_atom[i]:
            x.append(indis.featurize(df.structure[i],j))
    indis_mean.append(np.mean(x,0))
for i in range(0,len(labels)):
    indis_fea=[x[i] for x in indis_mean]
    df['MIG_'+labels[i]]=indis_fea

# OPSiteFingerprint
from matminer.featurizers.site import OPSiteFingerprint
opf=OPSiteFingerprint()
labels=opf.feature_labels()
opf_mean=[]
for i in range(len(df.structure)):
    x=[]
    for j in range(len(df.structure[i])):
        if df.structure[i][j].as_dict()['species'][0]['element'] == df.mot_atom[i]:
            x.append(opf.featurize(df.structure[i],j))
    opf_mean.append(np.mean(x,0))
for i in range(0,len(labels)):
    opf_fea=[x[i] for x in opf_mean]
    df['MIG_'+labels[i]]=opf_fea

# SiteElementalProperty
from matminer.featurizers.site import SiteElementalProperty
sep=SiteElementalProperty()
labels=sep.feature_labels()
sep_mean=[]
for i in range(len(df.structure)):
    x=[]
    for j in range(len(df.structure[i])):
        if df.structure[i][j].as_dict()['species'][0]['element'] == df.mot_atom[i]:
            x.append(sep.featurize(df.structure[i],j))
    sep_mean.append(np.mean(x,0))
for i in range(0,len(labels)):
    sep_fea=[x[i] for x in sep_mean]
    df['MIG_'+labels[i]]=sep_fea

# 2.3 Descriptors for the frame

# Transform structure to dictionary
from pymatgen import Structure
stru=[]
for i in range(len(df)):
    stru.append(Structure.as_dict(df.structure[i]))
df['structure_orig']=pd.Series(stru,index=df.index)

# Remove migrator
for i in range(len(df)):
    j=0
    while j < len(df.structure[i]):
        if df.structure[i][j].as_dict()['species'][0]['element'] == df.mot_atom[i]:
            df.structure[i].remove_sites([j])
        else:
            j+=1

# ChemicalOrdering
from matminer.featurizers.structure import ChemicalOrdering
cord=ChemicalOrdering(shells=(1, 2, 3), weight='area')
df=cord.featurize_dataframe(df,'structure',ignore_errors=True)

# MaximumPackingEfficiency
from matminer.featurizers.structure import MaximumPackingEfficiency
maxp=MaximumPackingEfficiency()
df=maxp.featurize_dataframe(df,'structure',ignore_errors=True)

# StructuralHeterogeneity
from matminer.featurizers.structure import StructuralHeterogeneity
shete=StructuralHeterogeneity(weight='area', stats=('minimum', 'maximum', 'range', 'mean', 'avg_dev'))
df=shete.featurize_dataframe(df,'structure',ignore_errors=True)

# MinimumRelativeDistances
from matminer.featurizers.structure import MinimumRelativeDistances 
mrd=MinimumRelativeDistances()
df=mrd.featurize_dataframe(df,'structure',ignore_errors=True)

#transform list to float
for i in range(len(df)):
    mean=sum(df['minimum relative distance of each site'][i])/len(df['minimum relative distance of each site'][i])
    df['minimum relative distance of each site'][i]=mean

# StructralComplexity
from matminer.featurizers.structure import StructuralComplexity as SC
sc=SC()
df=sc.featurize_dataframe(df,'structure',ignore_errors=True)

# 3. Store the data

#transform the structure to dict before write in excel
from pymatgen import Structure
for i in range(len(df)):
    df.structure[i]=Structure.as_dict(df.structure[i])
df.to_excel(excel_writer='../../data/Ea/M_Ea_LiMgBiOS.xlsx')

# 4. Predict Ea

# 4.1 Get data

import pandas as pd
import numpy as np
import pickle

df=pd.read_excel(io='../../data/Ea/M_Ea_LiMgBiOS.xlsx',index_col=0)
df=df.replace(np.inf,np.nan)
df.fillna(0,inplace=True)

columns=[i for i in np.load('../../data/Ea/labels.npy',allow_pickle=True) if i != 'BV_Ea']
X=df[columns].values
formula=np.array(df.formula)
icsd=np.array(df.index)

scaler=pickle.load(open('../../data/Ea/Ea_scaler.pkl','rb'))
model=pickle.load(open('../../data/Ea/Ea_model_rf.pkl','rb'))

X=scaler.transform(X)
y=model.predict(X)

# 4.2. Predict energy

from dscribe.descriptors import SOAP
import ase.io.vasp

species_type=np.load('../../data/E_and_f/Species_type.npy')
atoms=[]
for i in icsd:
    atoms.append(ase.io.vasp.read_vasp('../../data/E_and_f/optimization/%s/CONTCAR' % i))
soap=SOAP(rcut=8, nmax=5, lmax=5,species=species_type,average='outer')
output=soap.create(system=atoms,n_jobs=4)

e_scaler=pickle.load(open('../../data/E_and_f/e_scaler.pkl','rb'))
e_pca=pickle.load(open('../../data/E_and_f/e_pca.pkl','rb'))
e_model=pickle.load(open('../../data/E_and_f/e_model.pkl','rb'))

e_X=e_scaler.transform(output)
e_X=e_pca.transform(e_X)
e_y=e_model.predict(e_X)

# 4.3. Store the data

from pymatgen import Composition
composition=[]
for compound in formula:
    composition.append(Composition(compound).fractional_composition)
    
elements=[]
fractions=[]
for com in composition:
    elements.append([str(i) for i in dict(com).keys()])
    fractions.append([str(i) for i in dict(com).values()])
    
frac_Li,frac_Mg,frac_Bi,frac_O,frac_S=[],[],[],[],[]
for i in range(len(elements)):
    frac_Li.append(fractions[i][elements[i].index('Li')])
    frac_Mg.append(fractions[i][elements[i].index('Mg')])
    frac_Bi.append(fractions[i][elements[i].index('Bi')])
    frac_O.append(fractions[i][elements[i].index('O')])
    frac_S.append(fractions[i][elements[i].index('S')])

Ef=[float(e_y[i])+float(frac_Li[i])*3.0896+float(frac_Mg[i])*1.9511+float(frac_Bi[i])*4.1823+float(frac_O[i])*4.9921+
    float(frac_S[i])*4.2098 for i in range(len(icsd))]

# store data
df=pd.DataFrame()
df['formula']=pd.Series(formula,dtype=str,index=icsd)
df['frac_Li']=pd.Series(frac_Li,dtype=float,index=icsd)
df['frac_Mg']=pd.Series(frac_Mg,dtype=float,index=icsd)
df['frac_Bi']=pd.Series(frac_Bi,dtype=float,index=icsd)
df['frac_O']=pd.Series(frac_O,dtype=float,index=icsd)
df['frac_S']=pd.Series(frac_S,dtype=float,index=icsd)
df['E']=pd.Series(e_y,dtype=float,index=icsd)
df['Ef']=pd.Series(Ef,dtype=float,index=icsd)
df['Ea']=pd.Series(y,dtype=float,index=icsd)
df.to_excel('../../data/Ea/optimization_LiMgBiOS.xls')

