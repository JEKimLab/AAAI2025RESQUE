# RESQUE: Quantifying Estimator to Task and Distribution Shift for Sustainable Model Reusability

This is the official implementation of **RESQUE: Quantifying Estimator to Task and Distribution Shift for Sustainable Model Reusability**

## Installation
All library versions are within `requirements.txt`. To install all the required dependencies and python version in an Ubuntu environment, execute the following command:

```bash
sudo apt update
sudo apt install python3.8
pip install -r requirements.txt
```

## RESQUE
To obtain RESQUE for distributional shift:
```bash
python3.8 ReSQuE/Noise/RNoise.py
  -mp MODEL_PATH/model.pkl \
  -rs RANDOM_SEED \
  -save SAVE_RESULTS_TO \
  -data DATASET \
  -layer_name LAYER \
  -n_tp NOISE_TYPE \
  -n_lvl LEVEL 
```

## Retraining
To retrain a model, run `NoiseRetraining/train_dir/retrain.py` in the following format with the path of the original model:
```bash
python3.8 NoiseRetraining/train_dir/retrain.py
  -mp PATH_TO_MODEL/model.pkl \
  -save PATH_TO_SAVE \
  -acc CUTOFF_ACCURACY \
  -rs RANDOM_SEED \
  -tlc TRANSFORMS_LR_CUTOFF \
  -d DATASET \
  -n_tp NOISE_TYPE \
  -n_lvl LEVEL
```

## Retraining configurations
`options` contains the JSON configurations format for learning rate schedule, training transformations, and cutoff plans. Edit the default values according to the desired training/testing scheme.

## Code Carbon Initialization
To initialize Code Carbon, for measuring energy and carbon emission, run the following command to setup the carbon tracker instance:
```bash
! codecarbon init
```
