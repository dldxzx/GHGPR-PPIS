# GHGPR-PPIS
GHGPR-PPIS is an innovative protein-protein interaction site prediction model that leverages the GraphHeat network and GPR technology.

# Requirements
    (1) python 3.6
    (2) torch-1.10.2+cu113
    (3) torchaudio-0.10.2
    (4) torchvision-0.11.3+cu113
    (5) dgl-1.0.2+cu113
    (6) cudatoolkit-10.1.168
    (7) numpy-1.19.5
    (8) pandas
    (9) scikit-learn-0.22.2
## Datasets
    (1)The files in "./Dataset" include the datasets used in this experiment(test_315-28.pkl, test_315.pkl,test_60.pkl, train_335.pkl, UBtest_31-6.pkl, UBtest_31.pkl)<br>
    (2)The files in "./Feature" include the features used in this experiment.Among them, 741(335+60+315+31) represents the number of protein in GraphPPIS, and 706(335-1+315-28+60+31-6) represents the number of protein files with 741 protein excluded. In this paper, 706 protein after exclusion are used as data sets.
## Usage
  The construction of the model is in the "model.py".<br>
  The models with trained parameters are put in the directory "./Model/" and the predicted results is 'GHGPR_PPIS_predict.pkl'
  You can run the train.py file to train GHGPR-PPIS and use the predcit.py file to see the predictive resluts. If you want to tune some hyper-parameters, you can change some values of hyper-parameters in config.py in utils folder.
  The other details can see the paper and the codes.

  
