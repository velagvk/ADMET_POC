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
df=sys.argv[2]

if torch.cuda.is_available():
  device = torch.device('cuda') 
else:
  device = torch.device('cpu')
dataset=preprocess_dataset(df)
model=MolecularGraphNeuralNetwork(N=5000, dim=64,layer_hidden=4, layer_output=10, dropout=0.45).to(device)
if input=='solubility':
  model.load_state_dict(torch.load(path_for_saved_models+ '/Solubility' + '/output' + '/model' + '/model.pth'))
elif input=='permeability':
  model.load_state_dict(torch.load(path_for_saved_models+ '/Permeability' +  '/output' + '/model' + '/model.pth'))
elif input=='lipophilicity':
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
prediction=Predict(model,batch_test)
predictions = pediction.predict(dataset)[1]
tester.save_predictions(predictions, file_predicted_result)

  
  
