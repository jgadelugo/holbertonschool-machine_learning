#!/usr/bin/env python3
"""
Train weights to play atari
"""
from PIL import Image
import numpy as np
import gym

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
# from rl.processors import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from keras import layers
import keras as K


class AtariProcessor(Processor):
    """ Processor for Atari """
    def process_observation(self, observation):
        """Process observations"""
        INPUT_SHAPE = (84, 84)
        # Assert dimension (height, width, channel)
        assert observation.ndim == 3
        # Retrieve image from array
        img = Image.fromarray(observation)
        # Resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        # Convert back to array
        processed_observation = np.array(img)
        # Assert input shape
        assert processed_observation.shape == (84, 84)
        # Save processed observation in experience memory (8bit)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """ Convert the batches of images to float32"""
        # Convert the batches of images to float32 datatype
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """ process rewards"""
        return np.clip(reward, -1., 1.)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
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
    # Last layer: no. of neurons corresponds to action space
    # Linear activation
    activation = layers.Dense(nb_actions, activation='linear')(layer)
    model = K.Model(inputs=inputs, outputs=activation)

    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    # Initialize the atari_processor() class
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=nb_actions,
                   policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # training
    dqn.fit(env,
            nb_steps=1000000,
            log_interval=100000,
            visualize=False,
            verbose=2)

    # save the final weights.
    dqn.save_weights('policy.h5', overwrite=True)
    # Save model
    model.save("policy_model.h5")
