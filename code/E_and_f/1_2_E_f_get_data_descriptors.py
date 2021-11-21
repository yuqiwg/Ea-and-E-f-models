#!/usr/bin/env python
# coding: utf-8
# Time: 2021/10/11
# Author: Wang Yuqi
# File: 1_2_E_f_get_data_descriptors.py
'''

Energy and force model
1. Gather data
2. Extract descriptors

'''


# 1. Get data

#import all the atoms from '../../data/E_and_f/raw_data' which contains vasprun files
import os
import numpy as np
import ase.io.vasp

def get_dirs(name):
    lines=[]
    for dirs in os.walk(name):
        lines.append(dirs)
    return lines[0][1]

atoms=[]
name='../../data/E_and_f/raw_data/'

log=open('../../data/E_and_f/get_data.log','w')
log.write('Begin loading data\n\n')
log.close()

for dirname in get_dirs(name):
    i=0
    while True:
        try:
            atoms.append(list(ase.io.vasp.read_vasp_xml('%s%s/vasprun.xml' % (name,dirname),index=i))[0])
            i+=1
        except:
            break
log=open('../../data/E_and_f/get_data.log','a')
log.write('Number of atoms: %d \n\n' % (len(atoms)))

#get energies, forces, species, species_type from atoms
energies=[struc.get_total_energy()/len(struc.get_atomic_numbers()) for struc in atoms]
forces=[struc.get_forces() for struc in atoms]
species=[struc.get_chemical_symbols() for struc in atoms]
species_type=[]
for specie in species:
    species_type.extend(specie)
species_type=list(set(species_type))
log.write('types of species: %s \n\n' % (str(species_type)))


# 2. Extract descriptors

# 2.1 Energy

#SOAP descriptors for energy
from dscribe.descriptors import SOAP
species_type=np.load('../../data/E_and_f/species_type.npy')
soap=SOAP(rcut=8, nmax=5, lmax=5,species=species_type,average='outer')
n_features=soap.get_number_of_features()
log.write('Number of soap features: %d \n' % (n_features))

output=soap.create(system=atoms)
log.write('Number of atoms for soap: %d \n\n' % (len(output)))


# 2.2 Force

#choose some atoms for force model
interval=10
f_atoms=atoms[::interval] 
log.write('Number of atoms for force model: %d \n' % (len(f_atoms)))

#transform atoms to structures for agni
import pymatgen
structures=[]
for atom in f_atoms:
    ase.io.vasp.write_vasp('../../data/E_and_f/POSCAR',atom,vasp5=True)
    structures.append(pymatgen.Structure.from_file('../../data/E_and_f/POSCAR'))
os.remove('../../data/E_and_f/POSCAR')
log.write('Number of structures for force model: %d \n\n' % (len(structures)))

#AGNI descriptors
from matminer.featurizers.site import AGNIFingerprints

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

log.write('Number of structures for agni: %d \n' % (len(v)))
log.write('Number of agni features: %d \n\n' % (len(v[0][0][0])))


# 3. Store data

np.save("../../data/E_and_f/Energy.npy",energies)
np.save("../../data/E_and_f/E_Species.npy",species)
np.save("../../data/E_and_f/E_SOAP.npy",output)
np.save("../../data/E_and_f/Force.npy",forces[::interval])
np.save("../../data/E_and_f/F_species.npy",species[::interval])
np.save("../../data/E_and_f/F_AGNI.npy",v)
np.save("../../data/E_and_f/Species_type",species_type)
log.write('Finish \n')
log.close()

