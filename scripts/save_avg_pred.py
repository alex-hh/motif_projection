import argparse, os

import h5py
import numpy as np

def save_avg_preds(experiment_name, dataset):
  RESULT_DIR = os.environ.get('RESULT_DIR', 'results')
  preds = np.load(RESULT_DIR + '/predictions-best/{}-{}_full.npy'.format(experiment_name, dataset))
  nforward = preds.shape[0] // 2
  scores = np.stack((preds[:nforward], preds[nforward:]), axis=1)
  scores = np.mean(scores, axis=1)
  outfilepath = RESULT_DIR + '/predictions-best/{}-{}_full.h5'.format(experiment_name, dataset)
  with h5py.File(outfilepath, 'w') as h5f:
    h5f['preds'] = scores

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', help='Name of the experiment.')
  parser.add_argument('dataset', nargs='?', default='valid', type=str)
  args = parser.parse_args()
  save_avg_preds(args.experiment, args.dataset)