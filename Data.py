####################################################################################################################
####################################################################################################################
# Prepare input data and parameters for deep learning model to identify transcription factor binding site.
# Author: Haiying Kong
# Last Modified: 12 July 2020
####################################################################################################################
####################################################################################################################
import numpy as np
import os
import shutil
import numpy.lib.recfunctions
import pickle
from collections import OrderedDict

####################################################################################################################
####################################################################################################################
# Set parameters.
data_dir = '/home/kong/Documents/Data/'
seq_length = 121
n_epochs = 100

####################################################################################################################
###################################################################################################################
# Read in sequences and their classs.
Seq_Class = np.genfromtxt(data_dir + 'all.data.processed.txt',
                          dtype=[('Seq','U200'), ('Class','i1')],
 	       	          delimiter='\t', names=True, unpack=True)

Seq = Seq_Class['Seq']
Class = Seq_Class['Class']
Class = abs(Class.astype('int32') - 2)

####################################################################################################################
# Trim or pad sequences and make them all length of seq_length nt.
for i in range(len(Seq)):
  seq_len = len(Seq[i])
  if seq_len < seq_length:
    pad_len = seq_length - seq_len
    if pad_len % 2 == 0:
      Seq[i] = 'N' * int(pad_len/2) + Seq[i] + 'N' * int(pad_len/2)
    if pad_len % 2 == 1:
      n = int(pad_len/2)
      Seq[i] = 'N' * n + Seq[i] + 'N' * (n+1)
  if seq_len > seq_length:
    del_len = seq_len - seq_length
    n = int(del_len/2)
    Seq[i] = Seq[i][n : (n + seq_length)]

# Represent the sequences with matrices by one hot spot encoding.
Seq = np.ndarray.tolist(Seq)
chars = 'ACGT'
n_chars = len(chars)

def one_hot_encode(seq):
  mapping = dict(zip("ACGTN", range(5)))    
  seq2 = [mapping[i] for i in seq]
  return np.eye(5)[seq2][:, 0:4]

Seq_Mat = np.zeros((len(Seq), seq_length, n_chars), dtype=np.uint8)

for i in range(len(Seq)):
  Seq_Mat[i, :, :] = one_hot_encode(Seq[i])


####################################################################################################################
# Save the results for all samples and all features.
data = OrderedDict()
data['Seq_Mat'] = Seq_Mat.astype('float16')
data['Class'] = Class

output = open(data_dir + 'data.pkl', 'wb')
pickle.dump(data, output)
output.close()


####################################################################################################################
####################################################################################################################
# Parameters.
params = OrderedDict()
params['seq_length'] = seq_length
params['conv_kernel_size'] = 11
params['conv_stride'] = 1
params['pool_size'] = 3
params['pool_stride'] = 1
params['dense_n_1'] = int((seq_length - (params['conv_kernel_size']-1) - 1) / params['conv_stride']) + 1
params['dense_n_1'] = int((params['dense_n_1'] - (params['pool_size']-1) - 1) / params['pool_stride']) + 1
params['dense_n_2'] = 10
params['dropout_rate'] = 0.2
params['reg_lambda'] = 0.5
params['optimizer'] = 'Adam'
params['learning_rate'] = 0.0001
params['n_epochs'] = n_epochs
output = open(data_dir + 'params1.pkl', 'wb')
pickle.dump(params, output)
output.close()

params = OrderedDict()
params['seq_length'] = seq_length
params['conv_kernel_size'] = 9
params['conv_stride'] = 1
params['pool_size'] = 3
params['pool_stride'] = 1
params['dense_n_1'] = int((seq_length - (params['conv_kernel_size']-1) - 1) / params['conv_stride']) + 1
params['dense_n_1'] = int((params['dense_n_1'] - (params['pool_size']-1) - 1) / params['pool_stride']) + 1
params['dense_n_2'] = 10
params['dropout_rate'] = 0.2
params['reg_lambda'] = 0.5
params['optimizer'] = 'Adam'
params['learning_rate'] = 0.0001
params['n_epochs'] = n_epochs
output = open(data_dir + 'params2.pkl', 'wb')
pickle.dump(params, output)
output.close()

params = OrderedDict()
params['seq_length'] = seq_length
params['conv_kernel_size'] = 7
params['conv_stride'] = 1
params['pool_size'] = 3
params['pool_stride'] = 1
params['dense_n_1'] = int((seq_length - (params['conv_kernel_size']-1) - 1) / params['conv_stride']) + 1
params['dense_n_1'] = int((params['dense_n_1'] - (params['pool_size']-1) - 1) / params['pool_stride']) + 1
params['dense_n_2'] = 20
params['dropout_rate'] = 0.3
params['reg_lambda'] = 0.5
params['optimizer'] = 'Adam'
params['learning_rate'] = 0.0001
params['n_epochs'] = n_epochs
output = open(data_dir + 'params3.pkl', 'wb')
pickle.dump(params, output)
output.close()


####################################################################################################################
####################################################################################################################
