

#!pip install pytdc
#!pip install deepchem
from data_preprocess import *
import pandas
from rdkit import Chem
from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch
"""
This function takes the input smiles string from the pandas dataframe and creates new columns with adjacency matrix, fingerprint vector.
"""
def preprocess_dataset(df):
  #df=df[df['weights']!=0]
  df['mol']=df['smiles'].apply(lambda x : Chem.AddHs(Chem.MolFromSmiles(x)))
  df['atoms']=df['mol'].apply(lambda x : create_atoms(x,atom_dict))
  df['molecular_size']=df['atoms'].apply(lambda x :len(x))
  df['i_jbond_dict']=df['mol'].apply(lambda x : create_ijbonddict(x, bond_dict))
  df['fingerprints']=df[['atoms','i_jbond_dict']].apply(lambda x :extract_fingerprints(radius,x['atoms'],x['i_jbond_dict'],fingerprint_dict,edge_dict),axis=1)
  df['adjacency']=df['mol'].apply(lambda x: Chem.GetAdjacencyMatrix(x))
  df['adjacency'] = df['adjacency'].apply(lambda x:torch.FloatTensor(x).to(device))
  df['fingerprints']=df['fingerprints'].apply(lambda x : torch.LongTensor(x).to(device))
  df['property']=df['property'].apply(lambda x : torch.FloatTensor([[float(x)]]).to(device))
  df['adjacnency_len']=df['adjacency'].apply(lambda x:len(x))
  df['fingerprints_len']=df['fingerprints'].apply(lambda x:len(x))
  df=df[df['adjacnency_len']==df['fingerprints_len']]
  dataset=list(zip(*map(df.get,['smiles', 'fingerprints','adjacency','molecular_size','property'])))
  return dataset








