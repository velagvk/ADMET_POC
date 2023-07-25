import pandas as pd
import torch
import matplotlib.pyplot as plt
import datetime
from GNN_GAN_model import *
from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error
import sys
folder=sys.argv[1]
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))   


def plot_training_loss(path_to_file_MAEs=folder):
    loss = pd.read_table(path_to_file_MAEs)
    
    loss['MAE_test']=loss['MAE_test'].str.replace(r'\[|\]','',regex=True).apply(lambda x:float(x))
    loss['MAE_train']=loss['MAE_train'].str.replace(r'\[|\]','',regex=True).apply(lambda x:float(x))
    plt.plot(loss['MAE_test'], color='r',label='MSE of test set')
#loss['MAE_test_v1']=loss['MAE_t'].apply(lambda x : x.detach().cpu().numpy())
#plt.plot(loss['MAE_dev'], color='b',label='MSE of validation set')
    plt.plot(loss['MAE_train'], color='y',label='MSE of train set')  
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    file_loss=path+'/output/'+time1+'-MAE.png'
    plt.savefig(file_loss,dpi=300)

#saving model and figures
def save_plot_model_predictions(path,dataset_train,dataset_test):
    torch.manual_seed(0)
    #model = MolecularGraphNeuralNetwork(
        #N, dim, layer_hidden, layer_output, dropout).to(device)
    #model.load_state_dict(torch.load(r'path'+ '/output/'+time1+'model'+'.h5'))
    model_path=path+ '/output' + '/model' + '/model.pth'
    model=torch.load(model_path)
    model.eval()
    time1=str(datetime.datetime.now())[0:13]
    file_test_result  = path+'/output/'+time1+ '_test_prediction'+ '.txt'
    file_train_result  = path+'/output/'+time1+ '_train_prediction'+ '.txt'
    #file_train_result  = path+'/output/'+time1+ '_train_prediction'+ '.txt'
    #file_model = path+ '/output_tf/'+time1+'_model'+'.h5'
    file1=path+'/output/'+time1+'-MAE.png'
    file2=path+'/output'+time1+'pc-train.png'
    file3=path+'/output'+time1+'pc-test.png'
    file4=path+'/output'+time1+'pc-val.png'
    tester=Tester(model,batch_test)
    predictions_train = tester.test_regressor(dataset_train)[1]
    tester.save_predictions(predictions_train, file_train_result )
    predictions_test = tester.test_regressor(dataset_test)[1]
    tester.save_predictions(predictions_test, file_test_result)
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    res_tf = pd.read_table(file_train_result)
    res_tf['Correct']=res_tf['Correct'].str.replace(r'\[|\]','',regex=True).apply(lambda x:float(x))
    res_tf['Predict']=res_tf['Predict'].str.replace(r'\[|\]','',regex=True).apply(lambda x:float(x))
    rmse=rmse(res_tf['Correct'],res_tf['Predict'])
    r2 = r2_score(res_tf['Correct'], res_tf['Predict'])
    mae = mean_absolute_error(res_tf['Correct'], res_tf['Predict'])
    medae = median_absolute_error(res_tf['Correct'], res_tf['Predict'])
    rmae = np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct']) * 100
    median_re = np.median(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    mean_re=np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    plt.plot(res_tf['Correct'], res_tf['Predict'], '.', color = 'blue')
    plt.plot([-8,3], [-8,3], color ='red')
    plt.ylabel('Predicted Property')  
    plt.xlabel('Experimental Property')        
    plt.text(-1,-3, 'R2='+str(round(r2,4)), fontsize=20)
    plt.text(0.5,2.75,'MAE='+str(round(mae,4)),fontsize=12)
    plt.text(-3, 2.75, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(-7.5, 2.75, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(-4.5, 1, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.text(-4.5, 2, 'rmse='+str(round(rmse,4)), fontsize=12)
    plt.savefig(file2,dpi=300)
    plt.show() 
  
    

