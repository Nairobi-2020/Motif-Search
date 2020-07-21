####################################################################################################################
####################################################################################################################
# Cross validation on the model trained from all data.
# Author: Haiying Kong
# Last Modified: 12 July 2020
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import pickle
import numpy as np
import os
import shutil
import copy
import random
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import DNA_Pattern
import Methods

####################################################################################################################
# Set parameters.
data_dir = '/home/kong/Documents/Personal_Add/Apple/Swiss/Hall_Heim/Project/Data/'
n_block = 10

####################################################################################################################
####################################################################################################################
# Run cross validation for each of 3 sets of parameters.
####################################################################################################################
####################################################################################################################

params_names = ['params1', 'params2', 'params3']
apple = OrderedDict()

for params_name in params_names:    \

  ##################################################################################################################
  # Clean up model_dir.
  model_dir = '/home/kong/Documents/Personal_Add/Apple/Swiss/Hall_Heim/Project/Lock/Models/' + params_name
  if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  os.mkdir(model_dir)    \

  ##################################################################################################################
  # Load parameters and data.
  pkl_file = open(data_dir + params_name + '.pkl', 'rb')
  params = pickle.load(pkl_file)
  pkl_file.close()    \

  pkl_file = open(data_dir + 'data.pkl', 'rb')
  data = pickle.load(pkl_file)
  pkl_file.close()    \

  ##################################################################################################################
  # Get indexes for cross validation.
  N = len(data['Class'])
  idx_all = list(range(N))
  random.shuffle(idx_all)
  block_size = int(N / n_block)    \

  ##################################################################################################################
  # Cross validation.
  ##################################################################################################################
  # List to save AUC.
  AUC = []    \

  # cross validation.
  for i_block in list(range(n_block)):    \

    # Get training and testing data.
    idx_test = idx_all[i_block*block_size : (i_block+1)*block_size]
    idx_train = list(np.delete(np.array(idx_all), idx_test))
    train = OrderedDict()
    test = OrderedDict()
    train['Seq_Mat'] = data['Seq_Mat'][idx_train, :]
    train['Class'] = data['Class'][idx_train]
    test['Seq_Mat'] = data['Seq_Mat'][idx_test, :]
    test['Class'] = data['Class'][idx_test]    \

    ################################################################################################################
    # Train the model with train data and save in tmp folder.
    trian_dna_pat = Methods.Train_Checkpoint(model_dir, params, train)

    # Test the model with test data.
    test_dna_pat = Methods.Testing(model_dir, params, test)

    # Compute AUC and add to the list.
    auc = roc_auc_score(test_dna_pat.labels, test_dna_pat.prob.numpy()[:,0])
    AUC.append(auc)

  # Get mean of AUC for the cross valiation.
  mean_AUC = sum(AUC) / len(AUC)
  apple[params_name] = mean_AUC

# Save the AUC results.
apple = np.array(list(apple.items()))

header = ['Params_ID', 'mean_AUC']
header = '\t'.join([str(x) for x in header])
np.savetxt('/home/kong/Documents/Personal_Add/Apple/Swiss/Hall_Heim/Project/Results/DeepLearning_CrossValidation_meanAUC.txt', apple, fmt='%s', delimiter='\t', header=header)


####################################################################################################################
####################################################################################################################

