# finger_inverse_kinematic

Implement finger inverse kinematic with  neural network and reinforcement learning algorithm ddpg.

<img src="https://github.com/qianyuez/finger_inverse_kinematic/blob/master/data/inverse_kinematic1.gif" width="350px"><img src="https://github.com/qianyuez/finger_inverse_kinematic/blob/master/data/inverse_kinematic2.gif" width="350px">


## Requirements
python3.6
- `numpy`
- `tensorflow`
- `opencv-python`


## Usage
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


## Train
`cd finger_inverse_kinematic`

To train regression model, run:

`python train.py --model_type 0`

To train ddpg model, run:

`python train.py --model_type 1`


## Test
`cd finger_inverse_kinematic`

To test regression model, run:

`python test.py --model_type 0`

To test ddpg model, run:

 `python test.py --model_type 1`
 
 
## Reference
The ddpg tensorflow implementation is copied from the reinforcement tutorial by MorvanZhou:
https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-DDPG/
