
## Running saved models

- The saved models (.pth) files are stored in the property/output/model/model.pth. The 'run_saved_model.py' file runs saved models and saves predictions in the form of text file. The predictions text file will be stored in the  property/output folder.

- Script for running the saved models

!python run_saved_model.py 'path_for_saved_models' 'path_for_csv_file' 'Property to predict'

Sample script for running the saved models

```
!python run_saved_model.py 'ADMET_GITHUB_Folder/saved_models' 'ADMET_GITHUB_Folder/sample_dataset/delaney-processed.csv' 'Solubility'

```
