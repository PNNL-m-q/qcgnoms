#script to build a database of XTB QC features. Takes in a openbabel pickle database of 3d tmol structures and inchi.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import shutil
import pandas as pd
import numpy as np
import subprocess
from openbabel import openbabel
import urllib
from scipy.linalg import eigh

in_pkl=sys.argv[1]
out_pkl=sys.argv[2]
mol_begin_slice = int(sys.argv[3])
mol_end_slice = int(sys.argv[4])


def k_ab(i,j,H):
    Hi=i*3
    Hj=j*3
    if i<j:
        Hi=j*3
        Hj=i*3
    K=H[Hi-3:Hi,Hj-3:Hj] 
    return K

def get_u_vec(atom0,atom1):
    M1_vec=np.array([atom0.GetX(),atom0.GetY(),atom0.GetZ()])
    M2_vec=np.array([atom1.GetX(),atom1.GetY(),atom1.GetZ()])
    T_vec=np.subtract(M1_vec,M2_vec)
    u_T_vec=np.divide(T_vec,np.linalg.norm(T_vec))
    return u_T_vec

def coord_to_data(xtbopt_coord, hess_file, wbo, fod, q):
    obconversion = openbabel.OBConversion()
    obconversion.SetInAndOutFormats("tmol", "tmol")
    mol = openbabel.OBMol()
    obconversion.ReadFile(mol,xtbopt_coord)
    #load the hessian file
    Natoms=mol.NumAtoms()
    flat_hessian=np.array([])
    hess = open(hess_file,"r")
    output = hess.readlines()
    for line in output:
        if '$' not in line:
            for value in line.split():
                flat_hessian=np.append(flat_hessian,float(value))
    # Build the hessian matrix
    hessian=np.zeros(((Natoms*3),(Natoms*3)))
    f_idx=0
    #needs to traverse the whole 3Nx3N matrix
    for i in range(len(hessian)):
        for j in range(len(hessian)):
            hessian[i][j]=flat_hessian[f_idx]
            f_idx=f_idx+1
    am=np.zeros(mol.NumAtoms())
    coord = np.empty( ( 0,3 ) )
    A=np.array([])
    B=np.array([])
    mol_bonds = openbabel.OBMolBondIter(mol)
    np_edge_attr=np.empty( (0,3) ) # matrix for edge attributes ingeter bond orders, Wieberg Bond Orders, and bond energy terms
    for bond in mol_bonds:
        idx1=bond.GetBeginAtomIdx()-1
        idx2=bond.GetEndAtomIdx()-1
        atom1=bond.GetBeginAtom()
        atom2=bond.GetEndAtom()
        A=np.append(A,idx1)
        B=np.append(B,idx2)
        A=np.append(A,idx2)
        B=np.append(B,idx1)
        coord = np.vstack( (coord, np.array([atom1.GetX(),atom1.GetY(),atom1.GetZ()]) ) )
        coord = np.vstack( (coord, np.array([atom2.GetX(),atom2.GetY(),atom2.GetZ()]) ) )
        am[idx1] = np.array(atom1.GetAtomicMass())
        am[idx2] = np.array(atom2.GetAtomicMass())
        ob_bo = bond.GetBondOrder()
        # clculate bond force constant
        u_AB=get_u_vec(atom1,atom2)
        K_ij=-k_ab(idx1+1,idx2+1,hessian)
        eig=eigh(K_ij)
        eigval=eig[0]
        eigvec=eig[1]
        k=0
        for i in range(3):
            k=k+eigval[i]*np.abs(np.dot(u_AB,eigvec[:,i]))
        for bo in wbo:
            if bo[0]-1 == idx1 and bo[1]-1 == idx2:
                np_edge_attr = np.vstack( (np_edge_attr,np.array([ob_bo,bo[2],k])) )
                np_edge_attr = np.vstack( (np_edge_attr,np.array([ob_bo,bo[2],k])) )
            if bo[0]-1 == idx2 and bo[1]-1 == idx1:
                np_edge_attr = np.vstack( (np_edge_attr,np.array([ob_bo,bo[2],k])) )
                np_edge_attr = np.vstack( (np_edge_attr,np.array([ob_bo,bo[2],k])) )
    np_edidx=np.vstack((A,B))
    #build the node feature matrix
    N_node_features=7 #node features: atomic_mass, fractional ocupation density, charges, x, y, z,collision_energy
    x_mat = np.empty( ( 0,N_node_features ) )
    for i in range(len(am)):
        n_f = np.zeros( N_node_features )
        n_f[0] = am[i] #node atomic mass
        n_f[1] = fod[i] #fod
        n_f[2] = q[i] # q
        n_f[3] = coord[i][0]
        n_f[4] = coord[i][1]
        n_f[5] = coord[i][2]
        #n_f[-1] = #equals the collision energy
        x_mat = np.vstack((x_mat,n_f))
    return x_mat, np_edidx, np_edge_attr

def xtb(structure):
    xtbf = open("xtb.out", "a+")
    coord = open("coord", "w")
    coord.write(structure)
    coord.close()
    subprocess.call(["xtb","--opt", "extreme", "--pop", "--wbo", "--fod", "coord"], stdout=xtbf, stderr=xtbf)
    subprocess.call(["xtb", "xtbopt.coord", "--ohess"], stdout=xtbf, stderr=xtbf)
    #if the xtb calculation crashes return none
    if os.path.exists('NOT_CONVERGED') is True:
        os.remove('NOT_CONVERGED')
        x, ei, ea = None, None, None
        return x, ei, ea
    if os.path.exists('charges') is False or os.path.exists('fod') is False or os.path.exists('hessian') is False or os.path.exists('xtbopt.coord') is False:
        x, ei, ea = None, None, None
        return x, ei, ea
    charges=np.loadtxt("charges")
    fod=np.loadtxt("fod")
    wbo=np.loadtxt("wbo")
    x, ei, ea = coord_to_data("xtbopt.coord","hessian",wbo,fod,charges)
    os.remove('xtbrestart')
    os.remove('xtbtopo.mol')
    os.remove('fod.cub')
    os.remove('coord')
    os.remove('charges')
    os.remove('fod')
    os.remove('wbo')
    os.remove('hessian')
    return x, ei, ea

df_xtb = pd.DataFrame(columns=('inchi', 'x', 'edge_index', 'edge_attr'))

df = pd.read_pickle(in_pkl)

N = len(df)

for i, row in df.iterrows():
    inchi = row['inchi']
    structure = row['structure']
    print("STATUS:", i, N, inchi)
    x, ei, ea = xtb(structure)
    if x is not None:
        df_xtb = df_xtb.append({'inchi':inchi, 'x':x, 'edge_index':ei, 'edge_attr':ea},ignore_index=True)
    else:
        print("ERROR XTB failed", i)
    #print(df_xtb)
    if i % 1000 == 0:
        df_xtb.to_pickle('qc'+str(i)+'.bkp')

df_xtb.to_pickle(out_pkl)
