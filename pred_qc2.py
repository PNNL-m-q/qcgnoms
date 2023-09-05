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

#load data
qc = pd.read_pickle(sys.argv[1])
CE = float(sys.argv[2])
weights = sys.argv[3]

#gnn_weights=sys.argv[3]

spec_vec_len=50000

def vec_to_spec(vec):
    mz=np.array([])
    i=np.array([])
    for idx,itn in enumerate(vec):
        mz=np.append(mz,idx/100)
        i=np.append(i,itn)
    return mz, i

def df_to_data(qc_data,collision_energy):
    am_idx=np.array([12.011, 15.999, 1.008, 14.007, 32.067, 28.086, 30.974, 35.453, 18.998, 126.904, 78.96, 74.922, 10.812, 79.904, 118.711])
    x=qc_data.x
    edge_index=qc_data.edge_index
    edge_attr=qc_data.edge_attr
    if len(x) == 1:
        x = x[0]
    if len(edge_index) == 1:
        edge_index = edge_index[0]
    if len(edge_attr) == 1:
        edge_attr = edge_attr[0]
    #rebuild x with the one hot encode
    oh_x = np.empty( ( 0,len(am_idx)+1 ) )
    for xi in x:
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
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #should be one attribute per edge
    if len(edge_index[0]) != len(edge_attr):
        return None
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

#construct the pred dataset
mols_to_pred=[]
for index, row in qc.iterrows():
    mol_graph=df_to_data(row, CE)
    mols_to_pred.append(mol_graph)

##Construct the GNN##
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch.nn import GLU
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=128, at_channels=128)
# model.load_state_dict(torch.load(gnn_weights))
# model.to(device)

#gnn_weights=["../weights/qc2_1.model"]
gnn_weights=[weights]

for i,t in enumerate(mols_to_pred):
    msout = open("pred_qc2.ms", "w")
    msout.write(qc.iloc[i]['smiles'] + " " +str(CE) +  '\n')
    pred_series=np.empty((0,50000))
    for gw in gnn_weights:
        model.load_state_dict(torch.load(gw,map_location=torch.device('cpu')))
        model.to(device)
        t.to(device)
        layer_list = model(t.x, t.edge_index, t.edge_attr,torch.tensor([0]).to(device) )
        pred = layer_list[-1]
        pred = pred.cpu()
        pred = pred.detach().numpy()
        pred = pred[0]
        pred=np.divide(pred,np.max(pred))
        pred[pred<0.0] = 0
        pred = pred**4
        #print(pred)
        pred_series=np.vstack((pred_series,pred))
    pred_mz, pred_i = vec_to_spec(np.average(pred_series,axis=0))
    plt.stem(pred_mz,pred_i, linefmt='r-',markerfmt=' ',basefmt=" ")
    plt.show()
    for j in range(len(pred_i)):
        msout.write(str(pred_mz[j]) +" "+ str(pred_i[j]) + '\n')
