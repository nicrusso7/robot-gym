"""
DDPG Agent learning example using Keras-RL.

Credits:
https://github.com/nplan/gym-line-follower/blob/master/examples/ddpg.py
"""

import os
import pickle

from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, Concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
import gym


# Number of past subsequent observations to take as input
from robot_gym.agents.ddpg.keras_rl_agent import DDPGAgent
from robot_gym.controllers.mpc import mpc_controller
from robot_gym.model.robots.ghost import ghost

window_length = 5


def build_agent(env):
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    print(env.observation_space.shape)

    # Actor model
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(128, activation="relu"))
    actor.add(Dense(128, activation="relu"))
    actor.add(Dense(64, activation="relu"))
    actor.add(Dense(nb_actions, activation="tanh"))
    actor.summary()
    # actor.

    # Critic model
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    critic.summary()

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    # Exploration policy - has a great effect on learning. Should encourage forward motion.
    # theta - how fast the process returns to the mean
    # mu - mean value - this should be greater than 0 to encourage forward motion
    # sigma - volatility of the process
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.5, mu=0.4, sigma=0.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    return agent


def train(env, dir_name, steps=25000, pretrained_path=None):
    agent = build_agent(env)
    # Load pre-trained weights optionally
    if pretrained_path is not None:
        agent.load_weights(pretrained_path)

    save_path = os.path.join("models", dir_name)
    os.makedirs(save_path, exist_ok=False)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=False)
    h = agent.fit(env, nb_steps=steps, visualize=False, verbose=2,
                  callbacks=[ModelIntervalCheckpoint(os.path.join(save_path, "checkpoints", "chkpt_{step}.h5f"),
                                                     interval=int(steps/20), verbose=1),
                             TensorBoard(log_dir=os.path.join("logs", dir_name))])

    pickle.dump(h.history, open(os.path.join(save_path, "history.pkl"), "wb"))

    agent.save_weights(os.path.join(save_path, "last_weights.h5f"), overwrite=True)


def test(env, path):
    agent = build_agent(env)
    agent.load_weights(path)
    agent.test(env, nb_episodes=20, visualize=False)


if __name__ == '__main__':
    args = {
        "robot_model": ghost.Ghost,
        "mark": "1",
        "controller_class": mpc_controller.MPCController,
        "debug": True,
        "policy": True,
        "render": True
    }
    env = gym.make("GoTo-v0", **args)
    # train(env, "ddpg_1", steps=100000, pretrained_path=None)
    test(env, "/home/nic/dev/robotics/playground/ddpg_vxwz2/20210522T203333-go/checkpoints/chkpt_6000000.h5f")
