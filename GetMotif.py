####################################################################################################################
####################################################################################################################
# Get PWM for motif.
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
import DNA_Pattern
import Methods

####################################################################################################################
# Set parameters.
model_dir = '/home/kong/Documents/Lock/Models/model1/'

####################################################################################################################
# Load parameters and data.
pkl_file = open(model_dir + 'params.pkl', 'rb')
params = pickle.load(pkl_file)
pkl_file.close()

####################################################################################################################
# Create template for W_b.
dna_pat = DNA_Pattern.DNA_Pattern(params)

# Load W_b at the last checkpoint of the best model.
checkpoint = tf.train.Checkpoint()
manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=3)
checkpoint.W_b = dna_pat.W_b
checkpoint.restore(manager.latest_checkpoint)

pwm = checkpoint.W_b['W_conv']
pwm = tf.squeeze(pwm).numpy()

# Save PWM.
header = ['A', 'C', 'G', 'T']
header = '\t'.join(header)
np.savetxt('/home/kong/Documents/Results/DeepLearning_PWM.txt', pwm, fmt='%s', delimiter='\t', header=header)

motif = []
for i in range(len(pwm)):
  idx = np.argmax(pwm[i,:])
  motif.append('ACGT'[idx])

motif = ''.join(motif)
print(motif)

####################################################################################################################
####################################################################################################################

