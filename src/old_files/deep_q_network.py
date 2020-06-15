import os
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from hyper_parameters import DISCOUNT_RATE, TAU, LEARNING_RATE, NUM_FRAMES


class DeepQ(object):
    """Constructs the desired deep q learning network"""

    def __init__(self, action_size, observation_size, lr=LEARNING_RATE):
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr = lr
        self.model = None
        self.target_model = None
        self.qvalue_evolution = []
        self.construct_q_network()

    def construct_q_network(self):
        self.model = Sequential()
        input_layer = Input(shape=(self.observation_size * NUM_FRAMES,))
        # norm_layer = BatchNormalization(axis=1)(input_layer)
        layer1 = Dense(self.observation_size * NUM_FRAMES)(input_layer)
        layer1 = Activation('relu')(layer1)
        # layer2 = Dense(self.observation_size)(layer1)
        # layer2 = Activation('relu')(layer2)
        layer3 = Dense(self.observation_size)(layer1)
        layer3 = Activation('relu')(layer3)
        layer4 = Dense(2 * self.action_size)(layer3)
        layer4 = Activation('relu')(layer4)
        output = Dense(self.action_size)(layer4)

        self.model = Model(inputs=[input_layer], outputs=[output])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        self.target_model = Model(inputs=[input_layer], outputs=[output])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        self.target_model.set_weights(self.model.get_weights())

    def predict_movement(self, data, epsilon):
        """ Predict movement of game controler where is epsilon probability randomly move.
        Returns the optimal action and the predicted reward for that action.
        """
        rand_val = np.random.random()
        q_actions = self.model.predict(data.reshape(1, self.observation_size * NUM_FRAMES), batch_size=1)

        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.action_size)
        else:
            opt_policy = np.argmax(np.abs(q_actions[0]))

        self.qvalue_evolution.append(q_actions[0][opt_policy])

        return opt_policy, q_actions[0][opt_policy]

    def predict_rewards(self, data):
        q_actions = self.model.predict(np.array(data).reshape(1, self.observation_size * NUM_FRAMES), batch_size=1)
        return q_actions[0]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch):
        """ Trains the network on a batch of input.
        The parameters are batches of states, actions, rewards, done booleans and next states. """
        batch_size = s_batch.shape[0]

        # Train according to the Bellman Equation
        targets = self.model.predict(s_batch, batch_size=batch_size)
        fut_action = self.target_model.predict(s2_batch, batch_size=batch_size)

        targets[:, a_batch.flatten()] = r_batch
        targets[d_batch, a_batch[d_batch]] += DISCOUNT_RATE * np.max(fut_action[d_batch], axis=-1)

        targets_ts = tf.convert_to_tensor(targets, dtype=tf.float32)
        loss = self.model.train_on_batch(s_batch, targets_ts)
        return loss

    def train_imitation(self, s_batch, t_batch):
        """ Trains network on generated data: Imitation Learning. """
        loss = self.model.train_on_batch(s_batch, t_batch)
        return loss

    def save_network(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save(os.path.join(path, 'network.h5'))
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(os.path.join(path, 'network.h5'))
        print("Successfully loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    def replace_target(self):
        """ The target network needs to be updated every specific number of timesteps. """
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)
