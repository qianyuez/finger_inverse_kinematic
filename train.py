import numpy as np
from env import Env
from regress_model import RegressModel
from ddpg import DDPG
import config
import argparse


def train():
    var = 5
    for e in range(epochs):
        state = env.reset()
        rewards = 0
        for step in range(steps):
            if model_type == 0:
                goal = state[-2:]
                loss = model.learn(state[np.newaxis, :], goal[np.newaxis, :])
                print('episode: {}/{}, loss: {}'.format(e, epochs, loss))
                action = model.choose_action(state)
                state, _, _ = env.step(action)
            if model_type == 1:
                action = model.choose_action(state)
                # add randomness to action selection for exploration
                action = np.clip(np.random.normal(action, var), -1, 1)
                next_state, reward, done = env.step(action)
                rewards += reward
                model.store_transition(state, action, reward, next_state)

                if model.memory_counter > model.memory_size:
                    # decay the action randomness
                    var *= .9995
                    model.learn()

                state = next_state

                if done:
                    print('episode: {}/{}, rewards: {}'.format(e, epochs, rewards))
                    break

            env.render()
        if e % 1000 == 0:
            model.save_model(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type', type=int, default=0,
        help='model type index, 0 for regression model and 1 for reinforcement learning model, default 0'
    )
    parser.add_argument(
        '--regress_path', type=str, default=config.REGRESSION_MODEL_PATH,
        help='path to save regression model weight file, default {}'.format(config.REGRESSION_MODEL_PATH)
    )
    parser.add_argument(
        '--rl_path', type=str, default=config.RL_MODEL_PATH,
        help='path to save ddpg model weight file, default {}'.format(config.RL_MODEL_PATH)
    )
    ARGS = parser.parse_args()

    model_type = ARGS.model_type
    env = Env(model_type=model_type, image_size=(350, 300))
    if model_type == 0:
        model = RegressModel(env, 2, 8)
        model_path = ARGS.regress_path
        model.load_model(model_path)
        epochs = 300000
        steps = 1
    elif model_type == 1:
        model = DDPG(2, 8, 1)
        model_path = ARGS.rl_path
        print(model_path)
        epochs = 20000
        steps = 300
    else:
        raise ValueError('unknown model type')

    train()
