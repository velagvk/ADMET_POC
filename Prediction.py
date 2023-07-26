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

import pandas as pd
import matplotlib.pyplot as plt



if torch.cuda.is_available():
  device=torch.device('cuda')
else:
  device=torch.device('cpu')


#This is the Graph Attention layer class, where the attention between each atoms are calcualted
class GraphAttentionLayer(nn.Module):
  def __init__(self,in_features,out_features,dropout,alpha,concat=True):
    super(GraphAttentionLayer,self).__init__()
    self.dropout=dropout
    self.concat=concat
    self.in_features=in_features
    self.out_features=out_features
    self.alpha=alpha
    self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
    self.a=nn.Parameter(torch.zeros(size=(2*out_features,1))) 
    torch.nn.init.xavier_uniform_(self.W,gain=2.0)
    self.leakyrelu=nn.LeakyReLU(self.alpha)

  def forward(self,input,adj):
    """
    input feature :[N,in_features] in_features no of elments in the input feature vector of the node
    adj:adjacency matrix of the graph dimension
    """ 
    h=torch.mm(input,self.W)
    N=h.size()[0]
    a_input=torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)
    e=self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))
    zero_vec=-9e10*torch.ones_like(e)
    attention=torch.where(adj>0,e,zero_vec)
    attention=F.softmax(attention,dim=1)
    attention=F.dropout(attention,self.dropout,training=self.training)
    h_prime=torch.matmul(attention,h)
    if self.concat:
      return F.elu(h_prime)
    else:
      return h_prime


class GAT(nn.Module):
  def __init__(self,nfeat,nhid,dropout,alpha,nheads):
    super(GAT,self).__init__()
    self.dropout=dropout
    self.attentions=[GraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(nheads)]
    for i,attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i),attention)
    self.out_att=GraphAttentionLayer(nhid,64,dropout=dropout,alpha=alpha,concat=False)
    self.nheads=nheads
  def forward(self,x,adj):
    x=F.dropout(x,self.dropout,training=self.training)
    z=torch.zeros_like(self.attentions[1](x,adj))
    for att in self.attentions:
      z=torch.add(z,att(x,adj))
    x=z/self.nheads
    x=F.dropout(x,self.dropout,training=self.training)
    x=F.elu(self.out_att(x,adj))
    return F.softmax(x,dim=1)



class MolecularGraphNeuralNetwork(nn.Module):
  def __init__(self,N_fingerprints,dim,layer_hidden,layer_output,dropout):
    super(MolecularGraphNeuralNetwork,self).__init__()
    self.layer_hidden=layer_hidden
    self.layer_output=layer_output
    self.embed_fingerprint=nn.Embedding(N_fingerprints,dim)
    self.W_fingerprint=nn.ModuleList([nn.Linear(dim,dim) for _ in range(layer_hidden)])
    self.W_output=nn.ModuleList([nn.Linear(dim,dim) for _ in range(layer_output)])
    self.W_property=nn.Linear(dim,1)
    self.dropout=dropout
    self.alpha=0.25
    self.nheads=2
    self.attentions=GAT(dim,dim,dropout,alpha=self.alpha,nheads=self.nheads).to(device)
  def pad(self,matrices,padvalues):
    shapes=[m.shape for m in matrices]
    M,N=sum(s[0] for s in shapes),sum(s[1] for s in shapes)
    zeros=torch.FloatTensor(np.zeros((M,N))).to(device)
    pad_matrices=padvalues+zeros
    i,j=0,0
    for k,matrix in enumerate(matrices):
      #print(k)
      m,n=shapes[k]
      pad_matrices[i:i+m,j:j+n]=matrix
      i+=m
      j+=n
    return pad_matrices
  def update(self,matrix,vectors,layer):
    hidden_vectors=torch.relu(self.W_fingerprint[layer](vectors))
    #print(len(hidden_vectors),'update')
    return hidden_vectors+torch.matmul(matrix,hidden_vectors)
  def sum(self,vectors,axis):
    sum_vectors=[torch.sum(v,0) for v in torch.split(vectors,axis)]
    return torch.stack(sum_vectors)
  def gnn(self,inputs):
    Smiles,fingerprints,adjacencies,molecular_sizes=inputs
    #print(len(adjacencies),len(fingerprints),'forward')
    fingerprints=torch.cat(fingerprints)
    adj=self.pad(adjacencies,0)
    #print(len(adj),len(fingerprints),'gnn')
    fingerprint_vectors=self.embed_fingerprint(fingerprints)
    #print(len(adj),len(fingerprint_vectors),'embed')
    for l in range(self.layer_hidden):
      hs=self.update(adj,fingerprint_vectors,l)
      fingerprint_vectors=F.normalize(hs,2,1)
    molecular_vectors=self.attentions(fingerprint_vectors,adj)
    molecular_vectors=self.sum(molecular_vectors,molecular_sizes)
    return Smiles,molecular_vectors

  def mlp(self,vectors):
    for i in range(self.layer_output):
      vectors=torch.relu(self.W_output[i](vectors))
    outputs=self.W_property(vectors)
    return outputs
  def forward_regressor(self,data_batch,train):
    inputs=data_batch[:-1]
    correct_labels=torch.cat(data_batch[-1])
    if train:
      
      Smiles,molecular_vectors=self.gnn(inputs)
      predicted_scores=self.mlp(molecular_vectors)
      a=nn.L1Loss()
      loss=a(predicted_scores,correct_labels)
      predicted_scores=predicted_scores.to('cpu').data.numpy()
      correct_labels=correct_labels.to('cpu').data.numpy()
      return Smiles,loss,predicted_scores,correct_labels
    else:
      with torch.no_grad():
        Smiles,molecular_vectors=self.gnn(inputs)
        predicted_scores=self.mlp(molecular_vectors)
        a=nn.L1Loss()
        loss=a(predicted_scores,correct_labels)
    predicted_scores=predicted_scores.to('cpu').data.numpy()
    correct_labels=correct_labels.to('cpu').data.numpy()

    return Smiles,loss,predicted_scores,correct_labels


class Trainer(object):
  def __init__(self,model,lr,batch_train):
    self.model=model
    self.batch_train=batch_train  
    self.lr=lr
    self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)
  def train(self,dataset):
    np.random.shuffle(dataset)
    N=len(dataset)
    loss_total=0
    SMILES,P,C='',[],[]
    SAE=0
    for i in range(0,N,self.batch_train):
      data_batch=list(zip(*dataset[i:i+1+self.batch_train]))
      Smiles,loss,predicted_scores,correct_labels=self.model.forward_regressor(data_batch,train=True)
      SMILES+=''.join(Smiles)+''
      P.append(predicted_scores)
      C.append(correct_labels)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      loss_total+=loss.item()
      SAE += sum(np.abs(predicted_scores-correct_labels))
    tru=np.concatenate(C)
    pre=np.concatenate(P)
    MAE=loss/N
    SMILES=SMILES.strip().split()
    #pred=[1 if i>0.15 else 0 for i in pre]
    predictions=np.stack((tru,pre))
    return MAE,loss_total, predictions


class Tester(object):
    def __init__(self, model,batch_test):
        self.model = model
        self.batch_test=batch_test
    def test_regressor(self, dataset):
        N = len(dataset)
        loss_total = 0
        SAE=0
        SMILES, P, C = '', [], []
        for i in range(0, N, self.batch_test):
            data_batch = list(zip(*dataset[i:i + self.batch_test]))
            (Smiles, loss, predicted_scores, correct_labels) = self.model.forward_regressor(
                data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            #loss_total += loss.item()
            P.append(predicted_scores)
            C.append(correct_labels)
            SAE += sum(np.abs(predicted_scores-correct_labels))
        
        MAE=SAE/N
        SMILES = SMILES.strip().split()
        tru = np.concatenate(C)
        pre = np.concatenate(P)
        loss=np.abs(pre-tru)
        #pred = [1 if i >0.15 else 0 for i in pre]
        #AUC = roc_auc_score(tru, pre)
        #cnf_matrix=confusion_matrix(tru,pred)
        #tn = cnf_matrix[0, 0]
        #tp = cnf_matrix[1, 1]
        #fn = cnf_matrix[1, 0]
        #fp = cnf_matrix[0, 1]
        #cc = (tp + tn) / (tp + fp + fn + tn)
        Tru=map(str,tru)
        Pre=map(str,pre)
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, Tru, Pre)])
        #predictions = np.stack((tru, pre))
        return  MAE, predictions

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')


def predict(path_for_saved_models,dataset,input):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    model=MolecularGraphNeuralNetwork(5000,64,4,10,0.45).to(device)
    if input=='Solubility':
      model.load_state_dict(torch.load(path_for_saved_models+ '/Solubility' + '/output' + '/model' + '/model.pth'))
    elif input=='Permeability':
      model.load_state_dict(torch.load(path_for_saved_models+ '/Permeability' +  '/output' + '/model' + '/model.pth'))
    elif input=='Lipophilicity':
      model.load_state_dict(torch.load(path_for_saved_models + '/Lipophilicity' + '/output' + '/model' + '/model.pth'))
    model.eval()
    tester = Tester(model,10)
    predictions_train = tester.test_regressor(dataset_train)[1]
    file_predicted_result  = path_for_saved_models+'/output/'+time1+ input+ '_prediction'+ '.txt'
    #file_train_result  = path+'/output/'+time1+ '_train_prediction'+ '.txt'
    #file_model = path+ '/output_tf/'+time1+'_model'+'.h5'
#file1=path+'/output/'+time1+'-MAE.png'
#file2=path+'/output/'+time1+'pc-train.png'
#file3=path+'/output/'+time1+'pc-test.png'
#file4=path+'/output/'+time1+'pc-val.png'
    try:  
      os.makedirs(path+ '/output/')
    except:  
      pass
    tester.save_predictions(predictions, file_predicted_result)
    return
    
   
