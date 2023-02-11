#script to build a database of Albafet features. Takes in a msms pickle database of smiles strings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import shutil
import pandas as pd
import numpy as np
from rdkit import Chem
from alfabet import model

in_pkl=sys.argv[1]
out_pkl=sys.argv[2]

bos={Chem.BondType.SINGLE:1.0, Chem.BondType.DOUBLE:2.0,Chem.BondType.TRIPLE:3.0,Chem.BondType.AROMATIC:1.5, Chem.BondType.UNSPECIFIED:0.0}

def smi_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    am=np.zeros(mol.GetNumAtoms())
    A=np.array([])
    B=np.array([])
    np_edge_attr=np.empty( (0,2) ) # matrix for edge attributes ingeter bond orders and bond dissocitaion energy
    try:
        result = model.predict([smiles])
    except KeyError:
        return None, None, None
    bond_idx = 0
    for bond in mol.GetBonds():
        idx1=bond.GetBeginAtomIdx()
        idx2=bond.GetEndAtomIdx()
        atom1=bond.GetBeginAtom()
        atom2=bond.GetEndAtom()
        if atom1.GetMass() == 1.008 or atom2.GetMass() == 1.008:
            continue
        A=np.append(A,idx1)
        B=np.append(B,idx2)
        A=np.append(A,idx2)
        B=np.append(B,idx1)
        am[idx1] = np.array(atom1.GetMass())
        am[idx2] = np.array(atom2.GetMass())
        ob_bo = bos[bond.GetBondType()]
        try:
            bde = result[result['bond_index'] == bond_idx].bde_pred.values[0]
        except IndexError:
            bde = 0
        np_edge_attr = np.vstack( (np_edge_attr,np.array([ob_bo,bde])) )
        np_edge_attr = np.vstack( (np_edge_attr,np.array([ob_bo,bde])) )
        bond_idx = bond_idx + 1
    np_edidx=np.vstack((A,B))
    #build the node feature matrix
    N_node_features=2 #node features: atomic_mass, collision_energy
    x_mat = np.empty( ( 0,N_node_features ) )
    for i in range(len(am)):
        n_f = np.zeros( N_node_features )
        n_f[0] = am[i] #node atomic mass
        x_mat = np.vstack((x_mat,n_f))
    return x_mat, np_edidx, np_edge_attr


df_abf = pd.DataFrame(columns=('smiles', 'x', 'edge_index', 'edge_attr'))

if in_pkl.endswith('.pkl'):
    df = pd.read_pickle(in_pkl)
elif in_pkl.endswith('.csv'):
    df = pd.read_csv(in_pkl)

df_smiles = pd.unique(df['Smiles'])

N = len(df)

for i, smiles in enumerate(df_smiles):
    if smiles is not None:
        print("STATUS:", i, smiles)
        x, ei, ea = smi_to_data(smiles)
        if x is not None:
            df_abf = df_abf.append({'smiles':smiles, 'x':x, 'edge_index':ei, 'edge_attr':ea},ignore_index=True)        

df_abf.to_pickle(out_pkl)
