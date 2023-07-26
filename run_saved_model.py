import pandas as pd
import torch
import matplotlib.pyplot as plt
import datetime
from GNN_GAN_model import *
import timeit
import sys
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.metrics import roc_auc_score,roc_curve
from Create_Data import *
import pandas as pd
import matplotlib.pyplot as plt
from tdc.single_pred import ADME
from plot_predictions import *
import sys
from Prediction import *
path_for_saved_models=sys.argv[1]
#dataframe of smiles string
path_for_data=sys.argv[2]
input=sys.argv[3]

if torch.cuda.is_available():
  device = torch.device('cuda') 
else:
  device = torch.device('cpu')
df=pd.read_csv(path_for_data)

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
def preprocess_dataset_1(df):
  if torch.cuda.is_available():
    device=torch.device('cuda')
  else:
    device=torch.device('cpu')
  #df=df[df['weights']!=0]
  df['mol']=df['smiles'].apply(lambda x : Chem.AddHs(Chem.MolFromSmiles(x)))
  df['atoms']=df['mol'].apply(lambda x : create_atoms(x,atom_dict))
  df['molecular_size']=df['atoms'].apply(lambda x :len(x))
  df['i_jbond_dict']=df['mol'].apply(lambda x : create_ijbonddict(x, bond_dict))
  df['fingerprints']=df[['atoms','i_jbond_dict']].apply(lambda x :extract_fingerprints(radius,x['atoms'],x['i_jbond_dict'],fingerprint_dict,edge_dict),axis=1)
  df['adjacency']=df['mol'].apply(lambda x: Chem.GetAdjacencyMatrix(x))
  df['adjacency'] = df['adjacency'].apply(lambda x:torch.FloatTensor(x).to(device))
  df['fingerprints']=df['fingerprints'].apply(lambda x : torch.LongTensor(x).to(device))
  #df['property']=df['property'].apply(lambda x : torch.FloatTensor([[float(x)]]).to(device))
  df['adjacnency_len']=df['adjacency'].apply(lambda x:len(x))
  df['fingerprints_len']=df['fingerprints'].apply(lambda x:len(x))
  df=df[df['adjacnency_len']==df['fingerprints_len']]
  dataset=list(zip(*map(df.get,['smiles', 'fingerprints','adjacency','molecular_size'])))
  return dataset
dataset=preprocess_dataset_1(df)

model=MolecularGraphNeuralNetwork(N=5000, dim=64,layer_hidden=4, layer_output=10, dropout=0.45).to(device)
if input=='Solubility':
  model.load_state_dict(torch.load(path_for_saved_models+ '/Solubility' + '/output' + '/model' + '/model.pth'))
elif input=='Permeability':
  model.load_state_dict(torch.load(path_for_saved_models+ '/Permeability' +  '/output' + '/model' + '/model.pth'))
elif input=='Lipophilicity':
  model.load_state_dict(torch.load(path_for_saved_models + '/Lipophilicity' + '/output' + '/model' + '/model.pth'))
model.eval()
time1=str(datetime.datetime.now())[0:13]
path=path_for_saved_models
file_predicted_result  = path+'/output/'+time1+ '_prediction'+ '.txt'
    #file_train_result  = path+'/output/'+time1+ '_train_prediction'+ '.txt'
    #file_model = path+ '/output_tf/'+time1+'_model'+'.h5'
#file1=path+'/output/'+time1+'-MAE.png'
#file2=path+'/output/'+time1+'pc-train.png'
#file3=path+'/output/'+time1+'pc-test.png'
#file4=path+'/output/'+time1+'pc-val.png'
prediction=Predict(model,10)
predictions = pediction.predict(dataset)[1]
tester.save_predictions(predictions, file_predicted_result)

  
  
