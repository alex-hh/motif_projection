import sys
import numpy as np

def make_predictions(model, X, fname, verbose=2):
  """
  Makes predictions on data array X using model and saves it to fname
  """
  print('Making predictions.')
  sys.stdout.flush()
  scores = model.predict(X, batch_size = 128, verbose=verbose) 
  np.save(fname, scores)
  return scores  