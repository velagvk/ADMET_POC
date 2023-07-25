
![Alt text] https://www.google.com/url?sa=i&url=https%3A%2F%2Fblog.atomwise.com%2Fbehind-the-ai-a-novel-approach-for-using-graph-neural-networks-to-improve-admet-predictions&psig=AOvVaw2YiHzQDzLiAU-qSq9nCpST&ust=1690347313768000&source=images&cd=vfe&opi=89978449&ved=0CBAQjRxqFwoTCNjokcuIqYADFQAAAAAdAAAAABAQ


ADMET Prediction using GNNs
The purpose of this POC is to predict the drug ADMET properties using the SMILES string as input. Given the SMILES string, the fingerprint vector is extracted from the SMILES string. Also, the adjacency matrix is extracted from the SMILES string.  It uses graph neural networks to predict drug ADMET properties. Here, we use graph attention and graph convolutioal network to obtain molecular vectors.The obtained vectors are then passed through a mlp layer for both regression and classification tasks.




