# finger_inverse_kinematic

# Introduction
Implement finger inverse kinematic with  neural network and reinforcement learning algorithm ddpg.


# Requirements
python3.6
- `numpy`
- `keras`
- `opencv-python`

# Usage
Use --help to see usage of train.py and test.py.
```
usage: train.py [-h] [--model_type MODEL_TYPE] [--regress_path REGRESS_PATH]
                [--rl_path RL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        model type index, 0 for regression model and 1 for
                        reinforcement learning model, default 0
  --regress_path REGRESS_PATH
                        path to save regression model weight file, default
                        ./models/regression_model/regression_model.ckpt
  --rl_path RL_PATH     path to save ddpg model weight file, default
                        ./models/rl_model/rl_model.ckpt
```

```
usage: test.py [-h] [--model_type MODEL_TYPE] [--regress_path REGRESS_PATH]
               [--rl_path RL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        model type index, 0 for regression model and 1 for
                        reinforcement learning model, default 0
  --regress_path REGRESS_PATH
                        path to load regression model weight file, default
                        ./models/regression_model/regression_model.ckpt
  --rl_path RL_PATH     path to load ddpg model weight file, default
                        ./models/rl_model/rl_model.ckpt
```
