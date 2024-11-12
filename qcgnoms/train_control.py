import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys

#load data
df = pd.read_pickle(sys.argv[1])
qc = pd.read_pickle(sys.argv[2])
# set the training variables
BATCH_SIZE=512
LEARNING_RATE=0.0001
EPOCHS=1000
JOB='ctrl'

print("JOB INFO", BATCH_SIZE,LEARNING_RATE, EPOCHS, JOB)

def df_to_data(row,qc_data):
    am_idx=np.array([12.011, 15.999, 1.008, 14.007, 32.067, 28.086, 30.974, 35.453, 2.01410178, 18.998, 126.904, 78.96, 74.922, 10.812, 79.904])
    x=qc_data.x.values
    edge_index=qc_data.edge_index.values
    edge_attr=qc_data.edge_attr.values
    #clean up missing or out of format data in each row
    if len(x) == 1:
        x = x[0]
    if len(edge_index) == 1:
        edge_index = edge_index[0]
    if len(edge_attr) == 1:
        edge_attr = edge_attr[0]
    if len(x) == 0:
        return None
    #rebuild x with the one hot encode from the atomic mass vector
    oh_x = np.empty( ( 0,len(am_idx)+1 ) )
    for xi in x:
        oh_xi = np.zeros(len(am_idx)+1)
        xi_am = xi[0]
        xi_am_idx = np.argwhere(am_idx == xi_am)
        oh_xi[xi_am_idx] = 1
        oh_x = np.vstack((oh_x,oh_xi))
    x = oh_x
    specvec=row['Vec_Spec']
    #collision energy must be in eV as float.
    f_ce=float(row['Collision_energy'])
    #filter: collision energy ranges in eV
    ce_min=30.0
    ce_max=40.0
    if f_ce < ce_min:
        return None
    if f_ce > ce_max:
        return None
    #set the collision energy feature in x
    for i in range(len(x)):
        x[i][-1] = f_ce
    try:
        x = torch.tensor(x, dtype=torch.float)
    except TypeError:
        return None
    #convert the numpy arrays to torch tensors for training.
    y = torch.tensor(specvec, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #sanity check: should be one attribute per edge
    if len(edge_index[0]) != len(edge_attr):
        return None
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

#select data in the MSMS database for training.
in_df = df

# Determine vector representation of the spectra
max_MZ = 50000 # max m/z value can be 500.00 m/z otherwise spectra will be truncated.
min_MZ = 0

MZ_vals = np.arange(min_MZ,max_MZ)
spec_length = len(MZ_vals)

MZ_lists = in_df['M/Z'].values
In_lists = in_df['Intensity'].values
n_dat = in_df.shape[0]

in_df['Vec_Spec'] = np.zeros(n_dat)
in_df['Vec_Spec'] = in_df['Vec_Spec'].astype(object)
in_df['MZ_vals'] = np.zeros(n_dat)
in_df['MZ_vals'] = in_df['MZ_vals'].astype(object)

in_df.reset_index(inplace=True)

#build the vector representation of the spectra
for i in range(0,n_dat):
    mz = np.array(MZ_lists[i],dtype='float64')
    itn = np.array(In_lists[i],dtype='float64')
    max_itn = np.max(itn)
    new_vec = np.zeros(spec_length)
    for j in range(0,len(mz)):
        mz_idx = np.where(np.round(mz[j]*100) == MZ_vals)
        new_vec[mz_idx] =  np.max((new_vec[mz_idx],itn[j]/max_itn))
    in_df.at[i,'Vec_Spec'] = new_vec
    in_df.at[i,'MZ_vals'] = MZ_vals
  
#construct the pyg dataset for training
mass=np.array([])
N_bonds=np.array([])
smiles=[]
dataset=[]
for index, row in in_df.iterrows():
    if row['InChI'] is not None:
        spec_vec_len = len(row['Vec_Spec'])
        qc_row = qc[qc['inchi'] == row['InChI']]
        mol_graph=df_to_data(row, qc_row)
        if mol_graph is not None:
            #print(row['InChI'])
            #print(mol_graph.edge_attr)
            #print(row['Vec_Spec'])
            dataset.append(mol_graph)
            smiles.append(row['InChI'])
            mass=np.append(mass, float(row['MW']))
            N_bonds=np.append(N_bonds, len(mol_graph.edge_index[0])/2)
            

#Modified to robustly partition the train and test set by inchi string.
test_smi=[]
while len(test_smi) != 1000: #pullout unique smiles strings to use as a test set. 
    rand_smi=random.choice(smiles)
    if rand_smi not in test_smi:
        test_smi.append(rand_smi)

print(test_smi)
train=[]
test=[]
test_smiles=[]
for didx, data in enumerate(dataset):
    if smiles[didx] in test_smi:
       test.append(data)
       test_smiles.append(smiles[didx])
    else:
        train.append(data)


##Construct the GNN##
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch.nn import GLU
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool

#control model with no edge features
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, at_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.N_features = 16
        self.N_heads = 8

        self.at1 =  GATv2Conv(self.N_features, at_channels,   heads=self.N_heads)
        self.l1 = Linear(self.N_features, at_channels*self.N_heads)

        self.at2 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l2 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at3 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l3 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at4 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l4 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at5 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l5 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at6 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l6 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at7 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l7 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at8 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l8 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at9 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l9 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.at10 =  GATv2Conv(at_channels*self.N_heads, at_channels,   heads=self.N_heads)
        self.l10 = Linear(at_channels*self.N_heads, at_channels*self.N_heads)

        self.lin = Linear(at_channels*self.N_heads, spec_vec_len) 

    def forward(self, x, edge_index, batch):
        x = F.elu(self.at1(x, edge_index) + self.l1(x))
        x = F.elu(self.at2(x, edge_index) + self.l2(x))
        x = F.elu(self.at3(x, edge_index) + self.l3(x))
        x = F.elu(self.at4(x, edge_index) + self.l4(x))
        x = F.elu(self.at5(x, edge_index) + self.l5(x))
        x = F.elu(self.at6(x, edge_index) + self.l6(x))
        x = F.elu(self.at7(x, edge_index) + self.l7(x))
        x = F.elu(self.at8(x, edge_index) + self.l8(x))
        x = F.elu(self.at9(x, edge_index) + self.l9(x))
        x = F.elu(self.at10(x, edge_index) + self.l10(x))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

print("Full/Train/Test Size/SVL", len(dataset), len(train), len(test), spec_vec_len)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=128, at_channels=128)
model.to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = torch.nn.MSELoss()

def loss_fn(pred, target):
    cos_sim = -torch.nn.functional.cosine_similarity(pred,target)
    loss = torch.mean(cos_sim) + 10*F.mse_loss(pred,target)
    return loss

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        y_rs = torch.reshape(data.y, (int(len(data.y)/spec_vec_len), spec_vec_len))
        loss = 1-torch.mean(torch.nn.functional.cosine_similarity(out,y_rs))
        optimizer.zero_grad()  # Clear gradients.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    return loss

def eval():
    ### Evaluate the model ###
    scores=np.array([])
    for t in test:
        t.to(device)
        pred = model(t.x, t.edge_index,torch.tensor([0]).to(device) )
        pred = pred.cpu()
        t = t.cpu()
        score = torch.nn.functional.cosine_similarity(pred, t.y)
        scores = np.append(scores,score.detach().numpy()[0])
    return scores

for epoch in range(1, EPOCHS):
    train()
    train_acc = train()
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    if epoch%20==0: # evalulate the test set scores every 20 epochs
        scores=eval()
        print("TEST", epoch, np.average(scores))
    scheduler.step(train_acc)
    
scores=eval()
print("TEST", epoch, np.average(scores))

#save the test set scores and smiles strings to an output file
score_smile_file = open("score_smi"+JOB+".txt", "w")
for i in range(len(scores)):
    score_smile_file.write(str(scores[i])+" "+str(test_smiles[i])+"\n")
    
#save the training weights  
torch.save(model.state_dict(), "gnn"+JOB+".model")

exit()
