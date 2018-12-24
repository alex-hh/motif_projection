Implementation of experiments incorporating projection layer to simplify representation of motifs and improve performance in neural networks for regulatory genomics described in https://www.biorxiv.org/content/early/2018/09/10/412734

## Training Models

python scripts/run_experiments.py <experiment_name>

Experiment names with associated settings are defined in expset_settings:
 * expset_settings/500bp_slim.py - experiments with projection layer, dropout in 3 layer cnn with varying number of filters in first layer
 * expset_settings/final_crnn_slimpy - experiments with convolutional-recurrent neural networks using projection layer to improve on DanQ performance on DeepSEA dataset.

e.g. to run the convolutional recurrent model with 320 first layer filters

python scripts/run_experiments.py final_crnn_320_300rd20
