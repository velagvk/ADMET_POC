import torch
import pandas as pd
import train
import predict
import numpy as np
import rdkit
import train
import plot_predictions
import sys
path=sys.argv[1]
def run_and_plot(path):
    path=path
    train.train(radius=1, dim=64, layer_hidden=4, layer_output=10, dropout=0.45, batch_train=8,batch_test=8, lr=1e-4, lr_decay=0.85, decay_interval=25, iteration=140, N=5000,path=path)  
