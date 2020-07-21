####################################################################################################################
####################################################################################################################
# Define methdds for DNA_Pattern class.
# Author: Haiying Kong
# Last Modified: 6 July 2020
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import numpy as np
import pickle
import copy
import DNA_Pattern

####################################################################################################################
####################################################################################################################
# Define gradients.
def Gradients(dna_pat, data):
  with tf.GradientTape(watch_accessed_variables=True) as tape:
    dna_pat(data)
    grads = tape.gradient(dna_pat.loss, dna_pat.trainable_variables)
  return grads

####################################################################################################################
# Define training the model with training data and save checkpoint and final model.
def Train_Checkpoint(model_dir, params, data):    \

  dna_pat = DNA_Pattern.DNA_Pattern(params)    \

  # Define optimizer.
  if dna_pat.optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = dna_pat.learning_rate)
  if dna_pat.optimizer == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(learning_rate = dna_pat.learning_rate)    \

  # Define an empty list to save loss for all epoch.
  CrossEntropy = []    \

  # Define checkpoint and create template to save training variables..
  checkpoint = tf.train.Checkpoint(step = tf.Variable(0))
  checkpoint.W_b = dna_pat.W_b    \

  # Define checkpoint manager.
  manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=2)    \

  # Train the model.
  if manager.latest_checkpoint:
    print('Restored from {}'.format(manager.latest_checkpoint))
  else:
    print('Initializing from scratch.')    \

  for epoch in range(dna_pat.n_epochs):
    grads = Gradients(dna_pat, data)
    optimizer.apply_gradients(zip(grads, dna_pat.trainable_variables))    \

    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % 2 == 0:
      manager.save()
      print('Saved checkpoint for step {}: {}'.format(int(checkpoint.step), model_dir))
      print('loss {:1.3f}'.format(dna_pat.loss.numpy()))    \

    # Add loss of the epoch to the list.
    CrossEntropy.append(dna_pat.loss.numpy().item())    \

  CrossEntropy = np.array(CrossEntropy)
  dna_pat.CrossEntropy = CrossEntropy    \

  return dna_pat    \

####################################################################################################################
# Define prediction model.
def Testing(model_dir, params, data):    \

  # Load W_b at the last checkpoint of the best model.
  checkpoint = tf.train.Checkpoint()
  manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=3)
  dna_pat = DNA_Pattern.DNA_Pattern(params)
  checkpoint.W_b = dna_pat.W_b     # Get a template to restore variables.
  checkpoint.restore(manager.latest_checkpoint)    \

  # Compute all attributes.
  dna_pat(data)    \

  return dna_pat


####################################################################################################################
####################################################################################################################

