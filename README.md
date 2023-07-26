
![Model](https://github.com/velagvk/ADMET_POC/blob/main/ADMET.png)

## ADMET POC
ADMET Prediction using GNNs
- The purpose of this POC is to predict the drug ADMET properties using the SMILES string as input. 
- Given the SMILES string, the fingerprint vectors are extracted. Also, the adjacency matrix is extracted from the SMILES string using the RDKIT Library.  
MODEL DESCRIPTION
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


## Input dataset

The datasets are extracted from the TDC, an open source library (https://tdcommons.ai).


## RUN THE MODEL

* For running the model, you need the path for saving the model predictions and the dataset. The dataset is the name of the dataset available on the TDC commons.

Example script 
```
!python run_and_plot.py '/content/drive/MyDrive/ADMET_GITHUB_Folder' 'Lipophilicity_AstraZeneca'

```

* Here, the first input is the directory and the second input is the name of the dataset.

## List of Datasets used from TDC commons (SOURCE: 

|NAME| Dataset_Description| Task_Description | No of molecules|
|:------:|:--------------:|:-------------:| :-----------------:|
| Caco-2|The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.| Regression. Given a drug SMILES string, predict the Caco-2 cell effective permeability.| 906|
| Liophilicity_Astrazenca  | Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption. From MoleculeNet.| Regression. Given a drug SMILES string, predict the activity of lipophilicity.|4200
|HydrationFreeEnergy_FreeSolv| The Free Solvation Database, FreeSolv(SAMPL), provides experimental and calculated hydration free energy of small molecules in water. The calculated values are derived from alchemical free energy calculations using molecular dynamics simulations. From MoleculeNet.|Regression. Given a drug SMILES string, predict the activity of hydration free energy.|642
|Solubility_AqSolDB| Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.| Regression. Given a drug SMILES string, predict the activity of solubility.|9,982
|VDss_Lombardo| VDss (Volumn of Distribution at steady state), Lombardo et al.The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.|Regression. Given a drug SMILES string, predict the volume of distributon.|1,130 drugs






## Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)

[conda install -c conda-forge pytdc](https://tdcommons.ai)



## Citations


	@article{10.1093/bioinformatics/btac676,
	author = {Sun, Jinyu and Wen, Ming and Wang, Huabei and Ruan, Yuezhe and Yang, Qiong and Kang, Xiao and Zhang, Hailiang and Zhang, Zhimin and Lu, Hongmei},
	title = "{Prediction of Drug-likeness using Graph Convolutional Attention Network}",
	journal = {Bioinformatics},
	year = {2022},
	month = {10}
	}





