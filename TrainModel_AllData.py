####################################################################################################################
####################################################################################################################
# Train model with full data to classify TF binding DNA sequences.
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
import DNA_Pattern
import Methods

####################################################################################################################
# Set parameters.
data_dir = '/home/kong/Documents/Data/'
model_dir = '/home/kong/Documents/Lock/Models/model1/'

####################################################################################################################
# Clean the folder to save model.
if os.path.isdir(model_dir):
  shutil.rmtree(model_dir)

os.mkdir(model_dir)
os.mkdir(model_dir + 'tmp')

####################################################################################################################
# Load parameters and data.
pkl_file = open(data_dir + 'params3.pkl', 'rb')
params = pickle.load(pkl_file)
pkl_file.close()
params['n_epochs'] = 400

pkl_file = open(data_dir + 'data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

# Save params to model_dir.
output = open(model_dir + 'params.pkl', 'wb')
pickle.dump(params, output)
output.close()

####################################################################################################################
# Train the model with full data and save.
Methods.Train_Checkpoint(model_dir, params, data)


####################################################################################################################
####################################################################################################################

