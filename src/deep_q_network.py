# Credit Abhinav Sagar:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial
# Code under MIT license, available at:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial/blob/master/LICENSE
import numpy as np
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from hyper_parameters import NUM_FRAMES, DECAY_RATE, TAU

OBSERVATION_SIZE = 100
NUM_ACTIONS = 100


class DeepQ(object):
    """Constructs the desired deep q learning network"""

    def __init__(self, action_size, lr=1e-5, observation_size=OBSERVATION_SIZE):
        # It is not modified from  Abhinav Sagar's code, except for adding the possibility to change the learning rate
        # in parameter is also present the size of the action space
        # (it used to be a global variable in the original code)
        self.action_size = action_size
        self.observation_size = observation_size
        self.model = None
        self.target_model = None
        self.lr_ = lr
        self.qvalue_evolution = []
        self.construct_q_network()

    def construct_q_network(self):
        # replacement of the Convolution layers by Dense layers, and change the size of the input space and output space

        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        input_layer = Input(shape=(self.observation_size * NUM_FRAMES,))
        layer1 = Dense(self.observation_size * NUM_FRAMES)(input_layer)
        layer1 = Activation('relu')(layer1)
        layer2 = Dense(self.observation_size)(layer1)
        layer2 = Activation('relu')(layer2)
        layer3 = Dense(self.observation_size)(layer2)
        layer3 = Activation('relu')(layer3)
        layer4 = Dense(2 * NUM_ACTIONS)(layer3)
        layer4 = Activation('relu')(layer4)
        output = Dense(NUM_ACTIONS)(layer4)

        self.model = Model(inputs=[input_layer], outputs=[output])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.target_model = Model(inputs=[input_layer], outputs=[output])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.target_model.set_weights(self.model.get_weights())

    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # nothing has changed from the original implementation
        rand_val = np.random.random()
        q_actions = self.model.predict(data.reshape(1, self.observation_size * NUM_FRAMES), batch_size=1)

        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        else:
            opt_policy = np.argmax(np.abs(q_actions))

        self.qvalue_evolution.append(q_actions[0, opt_policy])

        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        # nothing has changed from the original implementation, except for changing the input dimension 'reshape'
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, self.observation_size * NUM_FRAMES), batch_size=1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, self.observation_size * NUM_FRAMES),
                                                   batch_size=1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)
        loss = self.model.train_on_batch(s_batch, targets)
        # Print the loss every 100 iterations.
        if observation_num % 100 == 0:
            print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        # nothing has changed
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)
