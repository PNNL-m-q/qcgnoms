#script to build a database of Albafet features. Takes in a msms pickle database of smiles strings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import shutil
import pandas as pd
import numpy as np
from rdkit import Chem

in_pkl=sys.argv[1]
out_pkl=sys.argv[2]

bos={Chem.BondType.SINGLE:np.array([1,0,0,0]), Chem.BondType.DOUBLE:np.array([0,1,0,0]),Chem.BondType.TRIPLE:np.array([0,0,1,0]),Chem.BondType.AROMATIC:np.array([0,0,0,1])}

def coord_to_data(inchi):
    mol = Chem.inchi.MolFromInchi(inchi)
    mol = Chem.AddHs(mol)
    am=np.zeros(mol.GetNumAtoms())
    A=np.array([])
    B=np.array([])
    np_edge_attr=np.empty( (0,4) )
    bond_idx = 0
    for bond in mol.GetBonds():
        idx1=bond.GetBeginAtomIdx()
        idx2=bond.GetEndAtomIdx()
        atom1=bond.GetBeginAtom()
        atom2=bond.GetEndAtom()
        A=np.append(A,idx1)
        B=np.append(B,idx2)
        A=np.append(A,idx2)
        B=np.append(B,idx1)
        am[idx1] = np.array(atom1.GetMass())
        am[idx2] = np.array(atom2.GetMass())
        ob_bo = bos[bond.GetBondType()]
        np_edge_attr = np.vstack( (np_edge_attr,ob_bo))
        np_edge_attr = np.vstack( (np_edge_attr, ob_bo))
        bond_idx = bond_idx + 1
    np_edidx=np.vstack((A,B))
    N_node_features=2 #node features: atomic_mass, collision_energy
    x_mat = np.empty( ( 0,N_node_features ) )
    for i in range(len(am)):
        n_f = np.zeros( N_node_features )
        n_f[0] = am[i] #node atomic mass
        x_mat = np.vstack((x_mat,n_f))
    return x_mat, np_edidx, np_edge_attr

df_abf = pd.DataFrame(columns=('smiles', 'x', 'edge_index', 'edge_attr'))

df = pd.read_pickle(in_pkl)
df_smiles = pd.unique(df['Smiles'])

N = len(df)

for i, smiles in enumerate(df_smiles):
    if smiles is not None:
        print("STATUS:", i, smiles)
        x, ei, ea = smi_to_data(smiles)
        if x is not None:
            df_abf = df_abf.append({'smiles':smiles, 'x':x, 'edge_index':ei, 'edge_attr':ea},ignore_index=True)        

df_abf.to_pickle(out_pkl)
