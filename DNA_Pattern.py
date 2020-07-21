####################################################################################################################
####################################################################################################################
# Define DNA_Pattern class. 
# Author: Haiying Kong
# Last Modified: 6 July 2020
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import numpy as np
import pickle
import copy

####################################################################################################################
####################################################################################################################
# Define the model.
class DNA_Pattern(tf.Module):    \

  ##################################################################################################################
  def __init__(self, params, name=None):
    super(DNA_Pattern, self).__init__(name=name)
    with self.name_scope:
      self.seq_length = params['seq_length']
      self.conv_kernel_size = params['conv_kernel_size']
      self.conv_stride = params['conv_stride']
      self.pool_size = params['pool_size']
      self.pool_stride = params['pool_stride']
      self.dense_n_1 = params['dense_n_1']
      self.dense_n_2 = params['dense_n_2']
      self.dropout_rate = params['dropout_rate']
      self.reg_lambda = params['reg_lambda']
      self.optimizer = params['optimizer']
      self.learning_rate = params['learning_rate']
      self.n_epochs = params['n_epochs']
      self.W_b = {
        'W_conv': tf.Variable(tf.random.truncated_normal([self.conv_kernel_size, 4, 1, 1], dtype=tf.float16), trainable=True, name='W_conv'),
        'W_dense_1': tf.Variable(tf.random.truncated_normal([self.dense_n_1, self.dense_n_2], dtype=tf.float16), trainable=True, name='W_dense_1'),
        'b_dense_1': tf.Variable(tf.zeros([self.dense_n_2], dtype=tf.float16), trainable=True, name='b_dense_1'),
        'W_dense_2': tf.Variable(tf.random.truncated_normal([self.dense_n_2, 2], dtype=tf.float16), trainable=True, name='W_dense_2'),
        'b_dense_2': tf.Variable(tf.zeros([2], dtype=tf.float16), trainable=True, name='b_dense_2')
        }    \

  ##################################################################################################################
  @tf.Module.with_name_scope
  def __call__(self, data):    \

    # Get input and output data.
    seq_mat = data['Seq_Mat']
    dim = list(seq_mat.shape)
    dim.append(1)
    seq_mat = tf.reshape(seq_mat, dim)
    clas = data['Class']    \

    # Set all dtype to float16.
    tf.keras.backend.set_floatx('float16')    \

    # Convolution:
    apple = tf.nn.conv2d(input=seq_mat, filters=self.W_b['W_conv'], strides=[1,self.conv_stride,4,1], padding='VALID', data_format='NHWC')
    apple = tf.squeeze(apple)    \

    # Normalization:
    mean_variance = tf.nn.moments(apple, axes=[0])
    apple = tf.nn.batch_normalization(x = apple, mean = mean_variance[0], variance = mean_variance[1],
                                      offset=None, scale=None, variance_epsilon=0.001)    \

    # Rectification:
    apple = tf.nn.relu(apple)    \
 
    # Pooling:
    dim = list(apple.shape)
    dim.append(1)
    apple = tf.reshape(apple, dim)
    max_pool_1d = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, strides=self.pool_stride, padding='valid', data_format='channels_last', dtype='float16')
    apple = max_pool_1d(apple)
    apple = tf.squeeze(apple)    \

    # Dropout:
    apple = tf.nn.dropout(apple, rate=self.dropout_rate)    \

    # Fully connected dense layer 1:
    apple = tf.matmul(apple, self.W_b['W_dense_1']) + self.W_b['b_dense_1']    \

    # Dropout:
    apple = tf.nn.dropout(apple, rate=self.dropout_rate)    \

    # Normalization:
    mean_variance = tf.nn.moments(apple, axes=[0])
    apple = tf.nn.batch_normalization(x = apple, mean = mean_variance[0], variance = mean_variance[1],
                                      offset=None, scale=None, variance_epsilon=0.001)    \

    # Rectification:
    apple = tf.nn.relu(apple)    \

    # Fully connected dense layer 2:
    apple = tf.matmul(apple, self.W_b['W_dense_2']) + self.W_b['b_dense_2']    \

    # Normalization:
    mean_variance = tf.nn.moments(apple, axes=[0])
    apple = tf.nn.batch_normalization(x = apple, mean = mean_variance[0], variance = mean_variance[1],
                                      offset=None, scale=None, variance_epsilon=0.001)    \

    # Centralize by row to make them logits.
    mean = tf.math.reduce_mean(apple, axis=1)
    logits = tf.math.subtract(apple, tf.stack([mean, mean], axis=1))    \

    # Add logits, labels and loss as attributes.
    self.logits = logits
    self.prob = tf.math.sigmoid(logits)
    self.pred = tf.math.argmax(self.logits, axis=1)
    self.labels = clas.astype('int32')
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.labels, self.logits))    \

    return self


####################################################################################################################
####################################################################################################################

