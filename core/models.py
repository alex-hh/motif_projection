from keras.optimizers import Adam
from keras.layers import Flatten, Dropout, Dense, Activation, MaxPooling1D,\
                         Bidirectional, LSTM
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model, Input, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D


class CNN():

  def __init__(self, l1_filters=300,
               sequence_length=1000,
               conv_dropout=0.2,
               lr=0.0003,
               embedding=True):
    self.sequence_length = sequence_length
    self.l1_filters = l1_filters
    self.conv_dropout = conv_dropout
    self.embedding = embedding
    self.lr = lr
  
  def get_compiled_model(self):
    inputs = Input(shape=(self.sequence_length, 4))
    x = Conv1D(self.l1_filters, 19, padding='valid')(inputs)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling1D(6, strides=6)(x)
    if self.embedding:
      x = Conv1D(64, 1, activation=None, use_bias=False)(x)
    x = Dropout(self.conv_dropout)(x)
    x = Conv1D(128, 11, padding='valid')(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling1D(2, strides=2)(x)
    x = Conv1D(256, 7, padding='valid')(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling1D(2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(2048, activation=None)(x)
    x = LeakyReLU(0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(919)(x)
    x = Activation('sigmoid')(x)
    self.model = Model(inputs, x)
    self.model.compile(optimizer=Adam(self.lr), loss='binary_crossentropy')
    return self.model


class CRNN():

  def __init__(self, filters=[320, 96],
               motif_embedding_dim=None,
               kernel_size=[30, 11],
               conv_dropout=[0., 0.],
               pooling_size=[7, 2],
               pooling_stride=[7, 2],
               sequence_length=1000,
               output_dim=919,
               recurrent=True,
               global_pooltype='mean',
               lr=0.0003,
               ncell=100,
               optimizer='adam',
               recurrent_dropout=0.,
               dense_dropout=0.,
               dense_units=[919]):
    self.filters = filters
    self.motif_embedding_dim = motif_embedding_dim
    self.sequence_length = sequence_length
    self.kernel_size = kernel_size
    self.conv_dropout = conv_dropout
    self.pooling_size = pooling_size
    self.output_dim = output_dim
    self.lr = lr
    self.pooling_stride = pooling_stride
    self.recurrent = recurrent
    self.global_pooltype = global_pooltype
    self.ncell = ncell
    self.optimizer = optimizer
    self.recurrent_dropout = recurrent_dropout
    self.dense_dropout = dense_dropout
    self.dense_units = dense_units
    self.model = None

  def apply_convolutions(self, x):
    for idx, params in enumerate(zip(self.filters, self.kernel_size, self.pooling_size, self.pooling_stride, self.conv_dropout)):
      f, k, p, p_st, cd = params
      x = Conv1D(f, k, padding='valid')(x)
      x = LeakyReLU(0.01)(x)
      x = MaxPooling1D(p, strides=p_st)(x)
      if idx == 0 and self.motif_embedding_dim:
        x = Conv1D(self.motif_embedding_dim, 1, activation=None, use_bias=False)(x)
      x = Dropout(cd)(x)
    return x

  def build_model(self):
    inputs = Input(shape=(self.sequence_length, 4))
    x = self.apply_convolutions(inputs)
    if self.recurrent:
      lstm_cell = LSTM(self.ncell, return_sequences=True,
                       recurrent_dropout=self.recurrent_dropout)
      x = Bidirectional(lstm_cell, merge_mode='concat')(x)
    if self.global_pooltype == 'mean':
      x = GlobalAveragePooling1D()(x)
    else:
      x = Flatten()(x)
    for units in self.dense_units:
      x = Dense(units)(x)
      x = LeakyReLU(0.01)(x)
      x = Dropout(self.dense_dropout)(x)
    output = Dense(self.output_dim, activation=None)(x)
    output = Activation('sigmoid')(output)
    self.model = Model(inputs, output)

  def get_compiled_model(self):
    loss = 'binary_crossentropy'
    if self.model == None:
      self.build_model()
    if self.optimizer == 'adam':
      self.model.compile(optimizer=Adam(self.lr), loss=loss)
    elif self.optimizer == 'sgd':
      self.model.compile(optimizer=SGD(lr=self.lr, momentum=0.9), loss=loss)
    else:
      raise ValueError('opimizer must be either adam or sgd')
    return self.model
