import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import shutil
import pandas as pd
import numpy as np
from rdkit import Chem
from alfabet import model

import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
import random
import sys
np.set_printoptions(threshold=sys.maxsize)
from numpy import dot
from numpy.linalg import norm
import networkx as nx

#GNN Dependencies#
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch.nn import GLU
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool

import pkg_resources

bos={Chem.BondType.SINGLE:1.0, Chem.BondType.DOUBLE:2.0,Chem.BondType.TRIPLE:3.0,Chem.BondType.AROMATIC:1.5, Chem.BondType.UNSPECIFIED:0.0}

spec_vec_len=50000

def smi_to_data(smiles, collision_energy):
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
        np_edidx=np.vstack((A,B))
        bond_idx = bond_idx + 1
        
    #build the node feature matrix
    N_node_features=2 #node features: atomic_mass, collision_energy
    x_mat = np.empty( ( 0,N_node_features ) )
    for i in range(len(am)):
        n_f = np.zeros( N_node_features )
        n_f[0] = am[i] #node atomic mass
        x_mat = np.vstack((x_mat,n_f))
        
    am_idx=np.array([12.011, 15.999, 1.008, 14.007, 32.067, 28.086, 30.974, 35.453, 18.998, 126.904, 78.96, 74.922, 10.812, 79.904, 118.711])
    #rebuild x with the one hot encode
    oh_x = np.empty( ( 0,len(am_idx)+1 ) )
    for xi in x_mat:
        oh_xi = np.zeros(len(am_idx)+1)
        xi_am = xi[0]
        xi_am_idx = np.argwhere(am_idx == xi_am)
        oh_xi[xi_am_idx] = 1
        oh_x = np.vstack((oh_x,oh_xi))
    x = oh_x
    
    f_ce=collision_energy
    #set the collision energy feature in x
    for i in range(len(x)):
        x[i][-1] = f_ce
    try:
        x = torch.tensor(x, dtype=torch.float)
    except TypeError:
        return None
    
    y = torch.tensor(0, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(np_edidx, dtype=torch.int64)
    edge_attr = torch.tensor(np_edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def vec_to_spec(vec):
    mz=np.array([])
    i=np.array([])
    for idx,itn in enumerate(vec):
        mz=np.append(mz,idx/100)
        i=np.append(i,itn)
    return mz, i

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, at_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.N_features = 16
        self.N_heads = 8

        self.at1 =  GATv2Conv(self.N_features, at_channels, edge_dim=2, heads=self.N_heads)
        self.l1 = Linear(self.N_features, at_channels*self.N_heads)

        self.at2 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l2 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at3 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l3 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at4 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l4 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at5 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l5 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at6 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l6 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at7 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l7 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at8 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l8 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at9 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l9 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at10 =  GATv2Conv(at_channels*self.N_heads, at_channels, edge_dim=2, heads=self.N_heads)
        self.l10 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.lin = Linear(at_channels*self.N_heads, spec_vec_len)

    def forward(self, x, edge_index, edge_attr, batch):
        x1  = F.elu(self.at1(x, edge_index, edge_attr) + self.l1(x))
        x2  = F.elu(self.at2(x1, edge_index, edge_attr) + self.l2(x1))
        x3  = F.elu(self.at3(x2, edge_index, edge_attr) + self.l3(x2))
        x4  = F.elu(self.at4(x3, edge_index, edge_attr) + self.l4(x3))
        x5  = F.elu(self.at5(x4, edge_index, edge_attr) + self.l5(x4))
        x6  = F.elu(self.at6(x5, edge_index, edge_attr) + self.l6(x5))
        x7  = F.elu(self.at7(x6, edge_index, edge_attr) + self.l7(x6))
        x8  = F.elu(self.at8(x7, edge_index, edge_attr) + self.l8(x7))
        x9  = F.elu(self.at9(x8, edge_index, edge_attr) + self.l9(x8))
        x10 = F.elu(self.at10(x9, edge_index, edge_attr) + self.l10(x9))
        x11 = global_mean_pool(x10, batch)
        x12 = self.lin(x11)
        return [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]
    
def concatenate_files(input_files, output_file, input_directory):
    with open(output_file, 'wb') as outfile:
        for filename in input_files:
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'rb') as infile:
                outfile.write(infile.read())

def predict_msms(smiles,collision_energy):
    
    weights_directory = pkg_resources.resource_filename('qcgnoms', 'weights')
    output_file = os.path.join(weights_directory, 'qc2_1.model')

    if not os.path.exists(output_file):
        input_files = [
            'qc2_1.model.aa', 'qc2_1.model.ab', 'qc2_1.model.ac',
            'qc2_1.model.ad', 'qc2_1.model.ae', 'qc2_1.model.af',
            'qc2_1.model.ag'
        ]
        concatenate_files(input_files, output_file, weights_directory)
    
    weights_path = pkg_resources.resource_filename('qcgnoms', 'weights/qc2_1.model')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(hidden_channels=128, at_channels=128)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.to(device)
    
    t = smi_to_data(smiles, collision_energy)
    t.to(device)
    layer_list = model(t.x, t.edge_index, t.edge_attr,torch.tensor([0]).to(device) )
    pred = layer_list[-1]
    pred = pred.cpu()
    pred = pred.detach().numpy()
    pred = pred[0]
    pred=np.divide(pred,np.max(pred))
    pred[pred<0.0] = 0
    
    pred_mz, pred_i = vec_to_spec(pred)
    
    return pred_mz, pred_i

