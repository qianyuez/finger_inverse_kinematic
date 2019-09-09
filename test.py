from env import Env, Point
from regress_model import RegressModel
from ddpg import DDPG
import cv2
import config
import argparse


def test():
    while True:
        if model_type == 0:
            action = model.choose_action(env.get_state())
            env.step(action)
            env.render(1)

        if model_type == 1:
            state = env.reset(reset_rotation=False)
            for step in range(300):
                action = model.choose_action(state)
                next_state, reward, done = env.step(action)
                state = next_state
                if done:
                    break
                env.render(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type', type=int, default=0,
        help='model type index, 0 for regression model and 1 for reinforcement learning model, default 0'
    )
    parser.add_argument(
        '--regress_path', type=str, default=config.REGRESSION_MODEL_PATH,
        help='path to load regression model weight file, default {}'.format(config.REGRESSION_MODEL_PATH)
    )
    parser.add_argument(
        '--rl_path', type=str, default=config.RL_MODEL_PATH,
        help='path to load ddpg model weight file, default {}'.format(config.RL_MODEL_PATH)
    )
    ARGS = parser.parse_args()

    model_type = ARGS.model_type
    env = Env(model_type=model_type, image_size=(350, 300))
    if model_type == 0:
        model = RegressModel(env, 2, 8)
        model.load_model(ARGS.regress_path)

        def on_mouse(event, x, y, flags, param):
            env.target_point = Point(x, y)
        cv2.namedWindow('screen')
        cv2.setMouseCallback('screen', on_mouse)
    elif model_type == 1:
        model = DDPG(2, 8, 1)
        model.load_model(ARGS.rl_path)
    else:
        raise ValueError('unknown model type')

    test()
