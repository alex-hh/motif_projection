import copy
from core.models import CRNN

expt_settings = []

crnn_base_settings = {'model_class': CRNN,
                      'model_args': {'kernel_size': [30, 11],
                                     'filters': [700, 96],
                                     'motif_embedding_dim': 64,
                                     'conv_dropout': [0.15, 0],
                                     'pooling_size': [7,2],
                                     'pooling_stride': [7,2],
                                     'lr': 0.0003,
                                     'ncell': 100,
                                     'recurrent': True,
                                     'global_pooltype': 'mean'},
                      'data_args': {'data_suffix': '_full'},
                      'callback_settings': {'es_patience': 5,
                                            'lr_patience': 2,
                                            'lr_factor': 0.2,
                                            'min_lr': 2e-6},
                      'train_settings': {'epochs': 35}}


l1s = [320, 700]
rds = [20]
lstm_sizes = [300]
# lstm_sizes = [200, 300, 500]

for l1 in l1s:
  for rd in rds:
    for l in lstm_sizes:
      # if l1 == 700:
        # sett['train_settings']['epochs'] = 1 # this would stop fit_generator...
      sett = copy.deepcopy(crnn_base_settings)
      sett['model_args']['filters'][0] = l1
      sett['model_args']['ncell'] = l
      sett['model_args']['recurrent_dropout'] = 0.01*rd
      # sett['model_args']['concat_pooled_conv'] = True
      sett['experiment_name'] = 'final_crnn_{}_{}rd{}'.format(l1, l, rd)
      expt_settings.append(sett)

      # if l1 == 700:
        # sett['train_settings']['epochs'] = 1
      sett = copy.deepcopy(crnn_base_settings)
      sett['model_args']['filters'][0] = l1
      sett['model_args']['ncell'] = l
      sett['model_args']['conv_dropout'] = [0.15, 0]
      sett['model_args']['motif_embedding_dim'] = None
      sett['model_args']['recurrent_dropout'] = 0.01*rd
      # sett['model_args']['concat_pooled_conv'] = True
      sett['experiment_name'] = 'final_crnn_{}_{}rd{}-control'.format(l1, l, rd)
      expt_settings.append(sett)
