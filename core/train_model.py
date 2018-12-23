import os, sys, pickle

from core.data import Data
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,\
                            LambdaCallback, CSVLogger


RESULT_DIR = os.environ.get('RESULT_DIR', 'results')

def train_model(model, data, callbacks, steps_per_epoch=None, verbose=2,
                epochs=30, class_weight=None, initial_epoch=0):  
  """
  Trains a model and uses callbacks to save weights.
  args:
      model - a compiled neural network
      data - a Data object which implements get_training_generator
      callbacks - A list of callbacks (which can be collected using the get_callbacks method)
      epochs - Defaulted to 5000 which will take days/weeks for networks I used i.e. for when I want a network to run
               until I cancel it or early stopping kicks in.
      class_weight - Weights the loss function, can be used to make some tasks contribute more/loss to the loss function
      verbose - 0 is nothing, 1 is full keras pct bar type thing, 2 is epoch end metrics
  """
  print('Training model for {} epochs'.format(epochs))
  print('model params: {}'.format(model.count_params()))
  print(model.summary())
  sys.stdout.flush()
  if steps_per_epoch is None:
    steps_per_epoch = data.steps_per_epoch()
  training_generator = data.get_training_generator()
  validation_data = data.get_data('valid')
  sys.setrecursionlimit(100000)  # required for very deep models see https://github.com/fchollet/keras/issues/2931#issuecomment-224504851
  model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks,
                      validation_data=validation_data, verbose=verbose, use_multiprocessing=True,
                      class_weight = class_weight, initial_epoch=initial_epoch)

def get_callbacks(experiment_name, es_patience=5, lr_patience=5000, lr_factor=0.5,
                  min_lr=0.0001, save_all_models=False, checkpoint=True):
  """
  args:
      es_patience - Early stopping patience i.e. how many epochs of not reducing validation loss before stopping
      lr_patience/lr_factor - number of epochs where validation loss doesn't decrease until lr reduces by lr_factor. 
                              Defaulted to 5000 to essentially be "off"
      min_lr - When using lr_patience and lr_factor the lower limit to the lr
      save_all_models - Saves model after every epoch instead of the best one. Models are large in size!
  """
  callbacks = []
  callbacks.append(CSVLogger('{}/csv_logs/{}.tsv'.format(RESULT_DIR, experiment_name), separator='\t'))
  callbacks.append(ModelCheckpoint('{}/models-best/{}.h5'.format(RESULT_DIR, experiment_name), save_best_only=True))
  if save_all_models:
    callbacks.append(ModelCheckpoint('{}/models-all/{}-{{epoch:02d}}.h5'.format(RESULT_DIR, experiment_name)))
  callbacks.append(EarlyStopping(patience=es_patience, verbose=1))
  callbacks.append(ReduceLROnPlateau(patience=lr_patience, factor=lr_factor, verbose=1, min_lr=min_lr))
  def flush(epoch, logs): return sys.stdout.flush()
  callbacks.append(LambdaCallback(on_epoch_begin=flush, on_epoch_end=flush))
  return callbacks

def get_untrained_model(experiment_name):
  settings = pickle.load(open('experiment_settings/{}.p'.format(experiment_name), 'rb'))
  model_class, model_args = settings['model_class'], settings['model_args'] 
  if 'reg_weighting' in model_args: 
    model_args['reg_weighting'] = get_weighting(**model_args['reg_weighting'])
  return model_class(**model_args)

def get_trained_model(experiment_name, epoch=None, use_gpu=True, return_class=False):
  if experiment_name == 'deepsea':
    from conservation.deepsea import DeepSea
    return DeepSea(use_gpu=use_gpu)
  else:
    model_class = get_untrained_model(experiment_name)
    model_class.get_compiled_model()
    model_class.model.load_weights('{}/models-best/{}.h5'.format(RESULT_DIR, experiment_name))
    if return_class:
      return model_class
    return model_class.model

def get_data_loader(experiment_name):
  data_args = pickle.load(open('experiment_settings/{}.p'.format(experiment_name), 'rb'))['data_args']
  return Data(**data_args)

