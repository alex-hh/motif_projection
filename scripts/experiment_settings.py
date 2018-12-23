import os, sys

from importlib import import_module
import pickle

from core.data import Data


def save_experiment_design(model_class, experiment_name, model_args={}, data_args={}, train_settings={}, 
                           callback_settings={}, check_model=False, check_data=False):
  to_save = {'model_class': model_class,
             'model_args': model_args,
             'data_args': data_args,
             'train_settings': train_settings,
             'callback_settings': callback_settings}
  if check_model:
    model_class(**model_args).get_compiled_model()
  if check_data:
    Data(**data_args)
  os.makedirs('experiment_settings', exist_ok=True)
  pickle.dump(to_save, open('experiment_settings/{}.p'.format(experiment_name), 'wb'))


def delete_experiment_design(experiment_name):
  os.remove('experiment_settings/{}.p'.format(experiment_name))


def print_experiments(experiments=None):
  if experiments is None:
    experiments = [x[:-2] for x in os.listdir('experiment_settings')]
  for experiment in sorted(experiments):
    print('===========', experiment, '===========')
    print_experiment(experiment)
    print()


def print_experiment(experiment_name):
  settings = pickle.load(open('experiment_settings/{}.p'.format(experiment_name), 'rb'))

  model_class, model_args, data_args = settings['model_class'], settings['model_args'], settings['data_args']
  train_settings, callback_settings = settings['train_settings'], settings['callback_settings']
  print('Model type : {}'.format(model_class.__name__))
  print('Model args :')
  for key in sorted(model_args.keys()):
    print('\t', key, model_args[key])
  print('Data args :')
  for key in sorted(data_args.keys()):
    print('\t', key, data_args[key])
  print('Train settings :')
  for key in sorted(train_settings.keys()):
    print('\t', key, train_settings[key])
  print('Callback settings :')
  for key in sorted(callback_settings.keys()):
    print('\t', key, callback_settings[key])


if __name__ == '__main__':
  assert len(sys.argv) >= 2
  which_exp_set = sys.argv[1]
  try:
    mod = import_module('expset_settings.' + which_exp_set)
    expt_settings = getattr(mod, 'expt_settings')
    for settings in expt_settings:
      save_experiment_design(**settings)
  except:
    raise Exception('Experiment name not recognised')
