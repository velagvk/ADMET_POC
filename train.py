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
folder=sys.argv[1]
import os
import plot_predictions as plt_pred


def train (radius, dim, layer_hidden, layer_output, dropout, batch_train,
    batch_test, lr, lr_decay, decay_interval, iteration, N ,path,data):

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
          path=path
          file_result = path + '/PREDICTIONS' + '.txt'
      #    file_result = '../output/result--' + setting + '.txt'
          result = 'Epoch\tTime(sec)\tLoss_train\tLoss_test\tprediction_train\tprediction_test'
          file_test_result = path + '\test_prediction' + '.txt'
          file_predictions = path + '\train_prediction' + '.txt'
          file_model = path+'/model' + '.pth'
          file_MAEs = path+ '/'+'MAEs'+'.txt'
          result_MAE  = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_test'
          with open(file_result, 'w') as f:
                f.write(result + '\n')

          lr =lr     #learning rate: 1e-5,1e-4,3e-4, 5e-4, 1e-3, 3e-3,5e-3
          lr_decay = lr_decay
          decay_interval = decay_interval
          iteration = iteration
          ratio=0.9
          iteration=iteration
          start = timeit.default_timer()


          file_result = path + '/PREDICTIONS' + '.txt'
                #    file_result = '../output/result--' + setting + '.txt'
          result = 'Epoch\tTime(sec)\tLoss_train\tLoss_test\tprediction_train\tprediction_test'
          file_test_result = path + '/test_prediction' + '.txt'
          file_predictions = path + '/train_prediction' + '.txt'
          file_model = path+'/model' + '.pth'
          file_MAEs = path+ '/'+'MAEs'+'.txt'
          result_MAE  = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_test'
          with open(file_MAEs , 'w') as f:
            f.write(result_MAE  + '\n')
          
          with open(file_result, 'w') as f:
                f.write(result + '\n')
          
          np.random.seed(1234)  # fix the seed for shuffle
          #np.random.shuffle(dataset)
          #n = int(ratio * len(dataset))
          #train_data = df.sample(frac = 0.75)
          #test_data=df.drop(train_data.index)
          #dataset_train=list(zip(*map(train_data.get,['smiles', 'fingerprints','adjacency','molecular_size','solubility'])))
          #dataset_test=list(zip(*map(test_data.get,['smiles', 'fingerprints','adjacency','molecular_size','solubility'])))
          if torch.cuda.is_available():
              device = torch.device('cuda')
          else:
              device = torch.device('cpu')
          model=MolecularGraphNeuralNetwork(N, dim,layer_hidden, layer_output, dropout)
          model=model.to(device)
          trainer = Trainer(model,lr,batch_train)
          tester = Tester(model,batch_test)


          for epoch in range(iteration):
                    epoch += 1
                    if epoch % decay_interval == 0:
                        trainer.optimizer.param_groups[0]['lr'] *= lr_decay
          
                    model.train()
                    # [‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]
                    MAE_train_v1, loss_train, train_res = trainer.train(dataset_train)
                    model.eval()
                    MAE_test,predictions = tester.test_regressor(dataset_test)
                    MAE_train, predictions= tester.test_regressor(dataset_train)
                    time = timeit.default_timer() - start
          
                    if epoch == 1:
                        minutes = time * iteration / 60
                        hours = int(minutes / 60)
                        minutes = int(minutes - 60 * hours)
                        with open(file_result, 'w') as f:
                          f.write(result + '\n')
                        print('The training will finish in about',
                              hours, 'hours', minutes, 'minutes.')
                        print('-' * 100)
                        print(result)
                    
                    
                    result = '\t'.join(map(str, [epoch, time, loss_train, MAE_train, MAE_test]))
                    tester.save_MAEs(result , file_MAEs )
                    
                    try:  
                        os.makedirs(path+ '/output/'+ 'model/')
                    except:  
                         pass
                    #print("Directory '% s' created" % file_model)
                    file_model = path+ '/output/'+ 'model/' + 'model' + '.pth'
                    tester.save_model(model, file_model)
                    print(result)
              
          plt_pred.save_plot_model_predictions(folder,dataset_train,dataset_test,N, dim, layer_hidden, layer_output, dropout,batch_train,batch_test)







