
![Model](https://github.com/velagvk/ADMET_POC/blob/main/ADMET.png)


ADMET Prediction using GNNs
The purpose of this POC is to predict the drug ADMET properties using the SMILES string as input. Given the SMILES string, the fingerprint vector is extracted from the SMILES string. Also, the adjacency matrix is extracted from the SMILES string.  It uses graph neural networks to predict drug ADMET properties. Here, we use graph attention and graph convolutioal network to obtain molecular vectors.The obtained vectors are then passed through a mlp layer for both regression and classification tasks.

|File| Description|
|:------:|:-------:|
| Create Data.py |  Takes the smiles string dataframe and applies the preprcessing function to extract fingerprint vectors and adjacency matrix  |
| GCN_GNN.py     |  Contains the model architecture
|data_preprocess.py| Contains the helper functions required for the Create Data.py
|train.py| Trains the model and creates the plots for the model predictions|
|run_and_plot.py| Takes the model parameters and passes it to the train.py. Also, saves the plot for the training and validataion loss.
|plot_predictions.py| Helper functions for generating and saving plots.




