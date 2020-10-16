#!/usr/bin/env python3
"""
Using training weights to play games of attari and show results
"""
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from keras import layers
import keras as K

AtariProcessor = __import__('train').AtariProcessor


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4

    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    # build Conv2d model
    inputs = layers.Input(shape=input_shape)
    perm = layers.Permute((2, 3, 1))(inputs)

    layer = layers.Conv2D(32, 8, strides=(4, 4), activation='relu',
                          data_format="channels_last")(perm)
    layer = layers.Conv2D(64, 4, strides=(2, 2), activation='relu',
                          data_format="channels_last")(layer)
    layer = layers.Conv2D(64, 3, strides=(1, 1), activation='relu',
                          data_format="channels_last")(layer)

    layer = layers.Flatten()(layer)
    layer = layers.Dense(512, activation='relu')(layer)
    # Linear activation
    activation = layers.Dense(nb_actions, activation='linear')(layer)
    model = K.Model(inputs=inputs, outputs=activation)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   processor=processor,
                   memory=memory)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # load weights.
    dqn.load_weights('policy.h5')

    # evaluate algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)
