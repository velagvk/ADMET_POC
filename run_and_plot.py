import torch
import pandas as pd
import train
import numpy as np
import rdkit
import train
import plot_predictions as plot_pred
import sys
folder=sys.argv[1]
dataname =sys.argv[2]
print(dataname)
train.train(radius=1, dim=64, layer_hidden=4, layer_output=10, dropout=0.45, batch_train=8,batch_test=8, lr=1e-4, lr_decay=0.85, decay_interval=25, iteration=50, N=5000,path=folder)  
plot_pred.plot_training_loss(folder)
