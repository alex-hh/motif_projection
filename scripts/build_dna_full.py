import h5py
import os
import numpy as np
from scipy.io import loadmat


SEQUENCE_LENGTH = 1000
NO_TASKS = 919

def create_dna(deepsea_path,
               dsets=['train', 'valid', 'test']):
  trainmat = h5py.File(deepsea_path + 'train.mat')
  validmat = loadmat(deepsea_path + 'valid.mat')
  testmat = loadmat(deepsea_path + 'test.mat')
  # validxdata, testxdata are (n_samples, 4, n_basepairs)
  # processed_data/dna['data_name'] are (n_samples, n_basepairs, 4)
  if 'train' in dsets:
    train_data = np.transpose(np.asarray(
        trainmat['trainxdata'], dtype='uint8'), axes=(2, 0, 1))
    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm]
    print('Data  with shape {} loaded. Writing to file'.format(train_data.shape), flush=True)

    with h5py.File('data/processed_data/dna_full.h5', 'a') as f:
      f.create_dataset('train', data=train_data, chunks=(32, SEQUENCE_LENGTH, 4))
      f.create_dataset('train_ids', data=perm)
  if 'valid' in dsets:
    valid_data = np.asarray(np.transpose(
        validmat['validxdata'], axes=(0, 2, 1)), dtype='uint8')
    with h5py.File('data/processed_data/dna_full.h5', 'a') as f:
      f.create_dataset('valid', data=valid_data, chunks=(32, SEQUENCE_LENGTH, 4))
  if 'test' in dsets:
    test_data = np.asarray(np.transpose(
        testmat['testxdata'], axes=(0, 2, 1)), dtype='uint8')
    with h5py.File('data/processed_data/dna_full.h5', 'a') as f:
      f.create_dataset('test', data=test_data, chunks=(32, SEQUENCE_LENGTH, 4))
  

def create_labels(deepsea_path,
                  dsets=['train', 'valid', 'test']):
  trainmat = h5py.File(deepsea_path+'train.mat')
  validmat = loadmat(deepsea_path+'valid.mat')
  testmat = loadmat(deepsea_path+'test.mat')

  label_dict = dict()

  if 'train' in dsets:
    perm = h5py.File('data/processed_data/dna_full.h5', 'r')['train_ids'][:]
    label_dict['train'] = np.asarray(trainmat['traindata'], dtype='uint8').T[perm]
    print(label_dict['train'].shape, flush=True)
  label_dict['valid'] = np.asarray(validmat['validdata'])
  label_dict['test'] = np.asarray(testmat['testdata'])

  with h5py.File('data/processed_data/labels_full.h5', 'w') as f:
    for dset in dsets:
      f.create_dataset(dset, data=label_dict[dset], chunks=(32, NO_TASKS))

if __name__ == '__main__':
  if not os.path.isfile('data/processed_data/dna_full.h5'):
    create_dna()
  create_labels()
