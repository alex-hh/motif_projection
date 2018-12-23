import os
import sys
import numpy as np
import h5py


class Data():
  """
  Object used to load data. The training data is returned batch by batch
  whereas the test and validation data the entire dataset is returned. A 
  generator can be returned for training using get_training_generator.
  
  get_data for 'valid' or 'train' returns the dataset in full.
  
  The data always returns the 4 DNA channels (A,C,G,T). Other arguments can add more channels  
  
  Args:
      batch_size - how many samples will be returned per iteration of the training generator
      shuffle - Shuffles the order batches are returned in each epoch, should always help convergance
                recommended to only set to False for debugging
      proportion_train_set_used - Used to subset the training data. For example if set to 0.2 only 20% of 
                                  the training data is used per epoch.
              
  """

  def __init__(self, batch_size=256, shuffle=True, proportion_train_set_used=1.0,
               proportion_val_set_used=1.0, data_suffix='_full', sequence_length=1000):
    self.seqstart_ind = (1000 - sequence_length) // 2
    self.seqend_ind = 1000 - self.seqstart_ind
    print('Loading data with length ', sequence_length)
    print(self.seqstart_ind)
    sys.stdout.flush()
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.proportion_train_set_used = proportion_train_set_used
    self.proportion_val_set_used = proportion_val_set_used
    self.train_index_generator = self.get_train_index_generator()

    if data_suffix is not None:
      self.suffix = data_suffix
    else:
      self.suffix = ''
    self.dna = h5py.File('data/processed_data/dna{}.h5'.format(self.suffix), 'r')
    self.labels = h5py.File('data/processed_data/labels{}.h5'.format(self.suffix), 'r')
    self.get_dataset_sizes()
    self.max_train_ind = (self.train_n // self.batch_size) * self.batch_size

  def get_dataset_sizes(self):
    self.train_n = int(self.proportion_train_set_used*self.dna['train'].shape[0])
    self.valid_n = int(self.proportion_val_set_used*self.dna['valid'].shape[0])
    print('N train {}, N val {}'.format(self.train_n, self.valid_n))
    print('Steps per epoch', self.steps_per_epoch())
    sys.stdout.flush()
    if 'test' in self.dna:
      self.test_n = self.dna['test'].shape[0]

  def get_training_generator(self, which_set='train'):
    while(True):
      try: 
        yield self.get_data(which_set)
      except Exception as e:
        print(e)
        print('Failed to get data')
        sys.stdout.flush()

  def get_data(self, dataset):
    """
    Retrieve the next batch of (inputs, targets), one-hot encoding on the fly

    Args:
      dataset (str): 'train', 'valid', or 'test'

    Returns: 
      inputs: ndarray (batch_size, SEQUENCE_LENGTH, num_channels)
      targets:
    """
    try:
      indexs = self.get_indexs(dataset) # get indices of next batch
    except:
      print('Failed to get indexes')
      sys.stdout.flush()
      self.train_index_generator = self.get_train_index_generator()
      indexs = self.get_indexs(dataset)

    Y = self.labels[dataset][indexs] # get labels of next batch by slicing? hdF5 file
    batch = []
    batch.append(self.dna[dataset][indexs])

    return np.concatenate(batch, axis=2)[:,self.seqstart_ind:self.seqend_ind,:], Y

  @staticmethod
  def reversecomp_dnaarr(dnaarr):
    """
    This is assuming dnaarr shape is n_samples, n_basepairs, input_dim
    """
    arr = np.zeros(dnaarr.shape)
    arr[:,:,:4] = np.rot90(dnaarr[:,:,:4], 2, axes=(2,1))
    arr[:,:,4:] = dnaarr[:,::-1,4:]
    return arr

  def get_indexs(self, dataset):
    if dataset == 'valid':
      indexs = list(range(self.valid_n))
    elif dataset == 'test':
      indexs = list(range(self.test_n))
    else:
      indexs = next(self.train_index_generator)
    return indexs

  def get_train_index_generator(self):
    """
    Return indices of datapoints in next batch.

    At each epoch, shuffle the order in which the batches are presented,
    but preserve the identities of the datapoints in each batch.
    i.e. - there are self.train_n // self.batch_size
      batches, which are always the same from epoch to epoch, but the order in which the
      batches are presented is shuffled
    """
    while True:
      # starts is calculated once per epoch
      self.starts = [x * self.batch_size for x in range(self.train_n//self.batch_size)]
      if self.shuffle:
        np.random.shuffle(self.starts) # this happens once per epoch
      for start in self.starts:
        # evaluated for each batch
        end = start + self.batch_size
        yield(range(start, end))

  def steps_per_epoch(self):
    """Number of times generator needs to be called to see entire dataset"""
    return self.train_n // self.batch_size
    
