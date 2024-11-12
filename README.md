# QC-GN<sup>2</sup>OMS<sup>2</sup> - A Graph Neural Net for ms/ms prediction.

## Prediction:
- Generate graph feature database in a pickle file
  - top_db.py : creates TOP feature database using RDKit
  - qc1_3d.py : generates 3D molecular structures for the QC1 database
  - qc1_db.py : aggregates the 3D structures and calculates features
  - qc2_db.py : calculates BDE features for the QC2 model and stores them in a pickle file
- Run the prediction script with the associated model and database
  - pred_afb.py : predict a spectra using the QC2 model


- Install:
```
git clone https://github.com/PNNL-m-q/qcgnoms.git
cd qcgnoms
mamba env create -f qcgnoms.yml
conda activate qcgnoms
pip install .
```

- Example:
```
$ python
>>> from qcgnoms import predict_msms
mz, itn = predict_msms("CC(C)/C=C/CCCCC(=O)NCC1=CC(=C(C=C1)O)OC", 45)
```

## Training:
### Requires a MS/MS database in a pickle file. Required data columns:
1. Collision Energy in eV
2. InChI
3. Smiles
4. M/Z: numpy array of high resolution m/z values.
5. Intensity: numpy array of MS intensities.
- See data/msms_sample.pkl

- Generate graph feature database in a pickle file
  - top_db.py : creates TOP feature database using RDKit
  - qc1_3d.py : generates 3D molecular structures for the QC1 database
  - qc1_db.py : aggregates the 3D structures and calculates features
  - qc2_db.py : calculates BDE features for the QC2 model and stores them in a pickle file
  
- Test datasets are assigned by first training the control model with train_control.py
- Test set data are located in test_set/

## Dependencies
- torch
- torch-geometric
- alfabet https://github.com/NREL/alfabet
- xtb https://github.com/grimme-lab/xtb
- openbabel
- pandas
- matplotlib
- numpy

## Citation
QC-GN<sup>2</sup>oMS<sup>2</sup>: a Graph Neural Net for High Resolution Mass Spectra Prediction <br />
Richard Overstreet, Ethan King, Julia Nguyen, Danielle Ciesielski
bioRxiv 2023.01.16.524269; doi: https://doi.org/10.1101/2023.01.16.524269 