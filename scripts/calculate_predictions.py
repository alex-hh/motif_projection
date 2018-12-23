import sys
import os
import argparse
from os.path import join

from core.evaluate_model import make_predictions
from core.data import Data
from core.train_model import get_trained_model, get_data_loader


sys.setrecursionlimit(100000)  # Needed for very deep models

RESULT_DIR = os.environ.get('RESULT_DIR', 'results')

def calculate_predictions(experiment_name, dataset):
  print('Loading data... ')
  sys.stdout.flush()
  if experiment_name in ['danq', 'deepsea', 'danqjaspar']:
    data = Data(data_suffix='_full')
    X, y = data.get_data(dataset)
  else:
    data = get_data_loader(experiment_name)
    X, y = data.get_data(dataset)

  print('Loading model... ')
  sys.stdout.flush()
  model = get_trained_model(experiment_name)
  print('Calculating predictions... ')
  sys.stdout.flush()
  make_predictions(model, X, join(RESULT_DIR, 'predictions-best',
                                  '{}-{}{}.npy'.format(experiment_name, dataset, data.suffix)),
                     verbose=1)

def make_dirs():
  os.makedirs('{}/predictions-best'.format(RESULT_DIR), exist_ok=True)
  os.makedirs('{}/predictions-all'.format(RESULT_DIR), exist_ok=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', help='Name of the experiment.')
  parser.add_argument('--dataset', help='Which dataset to make predictions on (valid or test)', default="test")
  args = parser.parse_args()
  print('Calculating predictions for {} on {}'.format(args.experiment, args.dataset))
  calculate_predictions(args.experiment, dataset=args.dataset)
