
![Model](https://github.com/velagvk/ADMET_POC/blob/main/ADMET.png)

## ADMET POC
ADMET Prediction using GNNs
- The purpose of this POC is to predict the drug ADMET properties using the SMILES string as input. 
- Given the SMILES string, the fingerprint vectors are extracted. Also, the adjacency matrix is extracted from the SMILES string using the RDKIT Library.  
- It uses graph neural networks to predict drug ADMET properties. 
* This model uses the graph attention and graph convolutioal networks to obtain molecular vectors.The obtained vectors are then passed through a mlp layer for both regression and classification tasks. 
* This model is inspired and some of the architecthure is taken from this paper (https://academic.oup.com/bioinformatics/article/38/23/5262/6759368)

## File Description

|File| Description|
|:------:|:-------:|
| Create Data.py |  Takes the smiles string dataframe and applies the preprcessing function to extract fingerprint vectors and adjacency matrix  |
| GCN_GNN.py     |  Contains the model architecture
|data_preprocess.py| Contains the helper functions required for the Create Data.py
|train.py| Trains the model and creates the plots for the model predictions|
|run_and_plot.py| Takes the model parameters and passes it to the train.py. Also, saves the plot for the training and validataion loss.
|plot_predictions.py| Helper functions for generating and saving plots.


## Usage




## Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)



## Citations


	@article{10.1093/bioinformatics/btac676,
	author = {Sun, Jinyu and Wen, Ming and Wang, Huabei and Ruan, Yuezhe and Yang, Qiong and Kang, Xiao and Zhang, Hailiang and Zhang, Zhimin and Lu, Hongmei},
	title = "{Prediction of Drug-likeness using Graph Convolutional Attention Network}",
	journal = {Bioinformatics},
	year = {2022},
	month = {10}
	}





