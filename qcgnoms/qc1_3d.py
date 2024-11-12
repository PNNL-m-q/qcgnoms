import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import shutil
import pandas as pd
import numpy as np
import subprocess
from subprocess import Popen
from openbabel import openbabel
import urllib

in_pkl=sys.argv[1]
out_pkl=sys.argv[2]

#takes in a MSMS db file containing all the 2d inchi strings and turns them into 3d tmol files for xtb

def nist_lookup(i,mol_str):
    #write the inchi string
    inchi_file="tmp/"+str(i)+".inchi"
    inchi = open(inchi_file, "w")
    inchi.write(mol_str)
    inchi.close()
    coord = "tmp/"+str(i)+".coord"
    if os.path.exists(coord) is False or os.stat(coord).st_size == 0:
        command=["obabel", "-i", "inchi", inchi_file, "-o", "tmol", "-O", coord,  "-h", "--gen3d"]
        return command

msms = pd.read_pickle(in_pkl)
inchis = pd.unique(msms['InChI'])

df = pd.DataFrame(columns=('inchi', 'structure'))

cmd_list = []

for i, mol in enumerate(inchis):
    if mol is not None:
        print("STATUS:", i, mol)
        command = nist_lookup(i,mol)
        if command is not None:
            cmd_list.append(command)

procs = [ Popen(i) for i in cmd_list ]
print("Running Procs")
for proc in procs_list:
	proc.wait()

def load(i):
    if os.path.exists(str(i)+".inchi") is False:
        return
    inchi = open(str(i)+".inchi", "r")
    inchi_str = inchi.readlines()
    if os.path.exists(str(i)+".coord") is False:
        return
    coord = open(str(i)+".coord", "r")
    return ' '.join(inchi_str), ' '.join(coord)

dfo = pd.DataFrame(columns=('inchi', 'structure'))

for i, mol in enumerate(inchis):
    if mol is not None:
        if load(i) is not None:
            inchi,coord = load(i)
            dfo = dfo.append({'inchi':inchi, 'structure':coord },ignore_index=True)

dfo.to_pickle(out_pkl)
