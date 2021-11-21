#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: Ea_gather_data_and_extract_descriptors.py
'''

Ionic migration activation energy model
1. Gather raw data
2. Extract descriptors

'''


# 1. Gathering raw data

# 1.1 Load the Ea dataset

#the file storing raw data
f=open('../../data/Ea/M_mig1','r')
# f=open('../../data/Ea/M_mig2','r')
# f=open('../../data/Ea/M_mig3','r')
# f=open('../../data/Ea/M_mig4','r')

E=[]
for line in f:
    E.append(line.split('\t'))
f.close()
icsd=[x[0] for x in E]
mp=[x[1] for x in E]
formula=[x[2] for x in E]
Ea=[x[3] for x in E]
ion=[x[4][:-1] for x in E]

#drop data without BV_Ea
i=0
while i < len(icsd):
    if Ea[i] == '[]':
        del icsd[i]
        del mp[i]
        del Ea[i]
        del ion[i]
        del formula[i]
    else:
        i+=1
Ea=[float(x) for x in Ea]


# 1.2 Add basic properties

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df=pd.DataFrame()

mot_val=[] #valence of migrating ion
mot_rad=[] #ionic radius of migrating ion
mot_ele=[] #electronegtivity of migrating ion
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
        
df['BV_Ea']=pd.Series(Ea,dtype=float,index=icsd)
df['mot_atom']=pd.Series(ion,index=icsd)
df['formula']=pd.Series(formula,index=icsd)
df['mot_val']=pd.Series(mot_val,dtype=float,index=icsd)
df['mot_rad']=pd.Series(mot_rad,dtype=float,index=icsd)
df['mot_ele']=pd.Series(mot_ele,dtype=float,index=icsd)

#add composition
from pymatgen import Composition
drop=[]
composition=[]
for i in range(len(df.index)):
    try:
        composition.append(Composition(df.formula[i]))
    except:
        drop.append(i)
for j in [df.index[i] for i in drop]:
    try:
        df.drop(labels=j,axis=0,inplace=True)
    except:
        continue
df['composition']=pd.Series(composition,index=df.index)

#add structure
import pymatgen as py
structure=[]
drop1=[]
for i in range(len(df.index)):
    try:
        structure.append(py.Structure.from_file('../../data/Ea/icsd_cif/%s.cif' % df.index[i])) #structure file
    except:
        drop1.append(i)
for j in [df.index[i] for i in drop1]:
    try:
        df.drop(labels=j,axis=0,inplace=True)
    except:
        continue
df['structure']=pd.Series(structure,index=df.index)
df.head()


# 2. Extracting descriptors

# 2.1 Descriptors for all the atoms

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

# 2.2 Descriptors for migrating atoms

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

# 2.3 Descriptors for frame atoms

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
df.to_excel(excel_writer='../../data/Ea/M_Ea1.xlsx')
# df.to_excel(excel_writer='../../data/Ea/M_Ea2.xlsx')
# df.to_excel(excel_writer='../../data/Ea/M_Ea3.xlsx')
# df.to_excel(excel_writer='../../data/Ea/M_Ea4.xlsx')
