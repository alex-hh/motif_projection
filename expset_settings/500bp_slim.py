import copy
from core.models import CNN

expt_settings = []

base_settings =  {'model_class': CNN,
                  'data_args': {'data_suffix': '_full',
                                'sequence_length': 500},
                  'callback_settings': {'es_patience': 5,
                                        'lr_patience': 2,
                                        'lr_factor': 0.2,
                                        'min_lr': 2e-6},
                  'model_args': {'l1_filters': 300,
                                 'conv_dropout': 0.2,
                                 'embedding': True,
                                 'lr': 0.0003,
                                 'sequence_length': 500},
                  'train_settings': {'epochs' : 35}}


l1_sizes = [300, 500, 750, 1000]
for l1_size in l1_sizes:
  sett = copy.deepcopy(base_settings)
  sett['model_args']['l1_filters'] = l1_size
  # suff += 'cd{}'.format(100*sett['model_args']['conv_dropout'])
  sett['experiment_name'] = 'cnn_slim_{}embedcd20'.format(l1_size)
  expt_settings.append(sett)

  sett = copy.deepcopy(base_settings)
  sett['model_args']['l1_filters'] = l1_size
  sett['model_args']['embedding'] = False
  sett['experiment_name'] = 'cnn_slim_{}cd20'.format(l1_size)
  expt_settings.append(sett)

  sett = copy.deepcopy(base_settings)
  sett['model_args']['l1_filters'] = l1_size
  sett['model_args']['embedding'] = False
  sett['model_args']['conv_dropout'] = 0.
  sett['experiment_name'] = 'cnn_slim_{}'.format(l1_size)
  expt_settings.append(sett)

  sett = copy.deepcopy(base_settings)
  sett['model_args']['l1_filters'] = l1_size
  sett['model_args']['conv_dropout'] = 0.
  sett['experiment_name'] = 'cnn_slim_{}embed'.format(l1_size)
  expt_settings.append(sett)
