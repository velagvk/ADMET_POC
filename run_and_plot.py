import torch
import pandas as pd
import train
import predict
import numpy as np
import rdkit
tes = train.train('../dataset/data_test.txt',   
    radius = 1,         
    dim = 52,           
    layer_hidden = 4,   
    layer_output = 10,  
    dropout = 0.45,    
    batch_train = 8, 
    batch_test = 8,     
    lr =3e-4,           
    lr_decay = 0.85,    
    decay_interval = 25, 
    iteration = 140,     
    N = 5000,          
    dataset_train='../dataset/data_train.txt')  
