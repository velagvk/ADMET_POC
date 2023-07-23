![image](https://github.com/velagvk/ADMET_POC/assets/139946537/44a372da-aade-4b91-8600-55111ca65322)


ADMET Prediction using GNNs


The purpose of this POC is to predict the drug ADMET properties using the SMILES string as input. Given the SMILES string, the fingerprint vector is extracted from the SMILES string. Also, the adjacency matrix is extracted from the SMILES string. 


uses graph neural networks to predict drug ADMET properties. Here, we use graph attention and graph convolutioal network to obtain molecular vectors.

The obtained vectors are then passed through a mlp layer for both regression and classification tasks.




