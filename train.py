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
data=ADME(name='Caco2_Wang')
split=data.get_split()
df_train=split['train']
df_test=split['test']
df_valid=split['valid']
df_train.rename(columns={'Drug':'smiles','Y':'property'},inplace=True)
df_test.rename(columns={'Drug':'smiles', 'Y':'property'},inplace=True)
df_valid.rename(columns={'Drug':'smiles','Y':'property'},inplace=True)



dataset_train=preprocess_dataset(df_train)
dataset_valid=preprocess_dataset(df_valid)
dataset_test=preprocess_dataset(df_test)


model=MolecularGraphNeuralNetwork(
          5000, 64, 4, 10, 0.45)




