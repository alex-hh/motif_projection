import os, sys, re
import pickle
import argparse

from core.train_model import train_model, get_callbacks, get_untrained_model
from core.data import Data

RESULT_DIR = os.environ.get('RESULT_DIR', 'results')

def create_directories():
  os.makedirs('results/csv_logs', exist_ok=True)
  os.makedirs('results/models', exist_ok=True)

def run_experiment(experiment_name):
  train_settings = get_training_settings(experiment_name)
  model = get_untrained_model(experiment_name).get_compiled_model()
  data = get_data(experiment_name)
  print('Model and data loaded')
  sys.stdout.flush()
  callbacks = get_callbacks(experiment_name, **get_callback_settings(experiment_name))
  train_model(model, data, callbacks, **train_settings)

def get_data(experiment_name):
  data_args = pickle.load(open('experiment_settings/{}.p'.format(experiment_name), 'rb'))['data_args']
  return Data(**data_args)
  
def get_callback_settings(experiment_name):
  return pickle.load(open('experiment_settings/{}.p'.format(experiment_name), 'rb'))['callback_settings']

def get_training_settings(experiment_name):
  return pickle.load(open('experiment_settings/{}.p'.format(experiment_name), 'rb'))['train_settings']


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment", help="Name of the experiment.")
  args = parser.parse_args()
  create_directories()
  run_experiment(args.experiment)


