import numpy as np
import os
import ase.io.vasp
import pymatgen
from pymatgen.core import Lattice, Structure
from ase.calculators.calculator import Calculator
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from dscribe.descriptors import SOAP
from matminer.featurizers.site import AGNIFingerprints

class Learn(Calculator):
    """Class for doing RandomForest calculations.
    """
    implemented_properties = ['energy', 'forces']

    nolabel = True

    def __init__(self, e_scaler=None,f_scaler=None,e_pca=None,f_pca=None,e_model=None,f_model=None,desc=None,species=None,**kwargs):
        Calculator.__init__(self, **kwargs)

        self.e_scaler=e_scaler
        self.f_scaler=f_scaler
        self.e_pca=e_pca
        self.f_pca=f_pca
        self.e_model=e_model
        self.f_model=f_model
        self.desc=desc
        self.species=species
        
 
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.e_features, self.f_features, idx_f = self.features(atoms, self.desc, self.species,self.e_scaler,self.f_scaler,self.e_pca,self.f_pca)
        self.energy=self.learn_e(self.e_model,self.e_features)
        self.forces=self.learn_f(self.f_model,self.f_features,idx_f)
        self.results['energy']=self.energy
        self.results['forces']=self.forces


    def learn_e(self,e_model,e_features):
        energy=e_model.predict(e_features)
        return energy

    def learn_f(self,f_model,f_features,idx_f):
        forces=np.zeros((len(self.atoms),3))
        for i in range(0,len(f_model)):
            f_i=f_model[i].predict(f_features[i])
            forces[idx_f[i]]=f_i
        return forces       

    def features(self,atoms,desc,species,e_scaler,f_scaler,e_pca,f_pca):
        soap=SOAP(rcut=8, nmax=5, lmax=5,species=species,average='outer')
        output=soap.create(system=atoms,n_jobs=4)
        
        ase.io.vasp.write_vasp('CONTCAR', atoms, label=None, direct=False, sort=None, symbol_count=None, long_format=True, vasp5=True, ignore_constraints=False, wrap=False)
        structures=[Structure.from_file('CONTCAR')]
        
        etas=[i for i in map(np.log,np.arange(0.8,16,0.2))][2::2][1:]
        n=len(etas)
        agni=AGNIFingerprints(directions=['x','y','z'],etas=etas)
        v=[]
        for i in range(len(structures)):
            x=[]
            #chemical symbols of sites in a structure
            sub_species=atoms.get_chemical_symbols()
            #chemical species in a structure
            types_specie=list(set(sub_species))
            j=0
            while j < len(structures[i]):
                x1=[]
                for k in range(len(species)):
                    if species[k] in types_specie:
                        structure_new=structures[i].copy()
                        sites=[m for m in range(len(sub_species)) if sub_species[m] != species[k] and m != j]
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
        
        D_prec_e=[output]

        idx_f=[]
        D_prec_f=[]
        chem_symbol_line=np.array(atoms.get_chemical_symbols())
        a,b=np.array(v).shape[-2:]
        for spe in species:
            idx=np.where(chem_symbol_line==spe)
            D_numpy=np.array(v)[0][idx[0]].reshape(len(idx[0]),a*b,order='F')
            idx_f.append(idx[0])
            D_prec_f.append(D_numpy)
        
        e_features=e_scaler.transform(D_prec_e)
        e_features=e_pca.transform(e_features)
        f_features=[]
        for i in range(0,len(f_scaler)):
            f=f_scaler[i].transform(D_prec_f[i])
            f=f_pca[i].transform(f)
            f_features.append(f)
        
        return e_features, f_features, idx_f

