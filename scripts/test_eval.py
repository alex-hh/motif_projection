import os, sys
from core.data import Data
from core.train_model import get_trained_model

def append_to_losses(expt_name, dataset, loss,
                     filename='final_losses_{}.csv'.format(sys.argv[2])):
    with open(filename, 'a') as f:
        f.write('{},{},{}\n'.format(expt_name, dataset, loss))

RESULT_DIR = os.environ.get('RESULT_DIR', 'results')

data = Data(sequence_length=int(sys.argv[2]), data_suffix='_full')
m = get_trained_model(sys.argv[1])
print('evaluating model', flush=True)
l = m.evaluate(*data.get_data('test'))
print('saving results', flush=True)
append_to_losses(sys.argv[1], 'test', l)