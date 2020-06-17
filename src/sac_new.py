# tf2.0 friendly
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numbers import Number
from collections import deque

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input, Concatenate
    import tensorflow.keras.optimizers as tfko
    from tensorflow.keras.models import load_model

from l2rpn_baselines_old.utils import BaseDeepQ  # Baseline Q network.
from l2rpn_baselines_old.utils.ReplayBuffer import ReplayBuffer
from sac_training_param import TrainingParamSAC


class SACNetwork(object):
    # TODO: Should all networks have the same optimizer learning rate/decay?
    # TODO: change trining function according to the non-stochastic policy (take argmax)...

    def _build_q_NN(self):
        """ Build Q networks as in the baseline SAC """
        input_states = Input(shape=(self.observation_size,))
        input_action = Input(shape=(self.action_size,))
        input_layer = Concatenate()([input_states, input_action])

        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(self.action_size*2)(lay2)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(1, activation='linear')(lay3)

        model = Model(inputs=[input_states, input_action], outputs=[advantage])
        model.compile(loss='mse', optimizer=tfko.Adam(lr=self.lr))
        return model

    def _build_value_NN(self):
        input_states = Input(shape=(self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(self.action_size*2)(lay2)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(self.action_size, activation='relu')(lay3)
        state_value = Dense(1, activation='linear')(advantage)

        model = Model(inputs=[input_states], outputs=[state_value])
        model.compile(loss='mse', optimizer=tfko.Adam(lr=self.lr))
        return model

    def _build_policy_NN(self):
        """ Build policy network as in the baseline SAC """
        input_states = Input(shape=(self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(self.action_size*2)(lay2)
        lay3 = Activation('relu')(lay3)

        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)

        model = Model(inputs=[input_states], outputs=[soft_proba])
        model.compile(loss='categorical_crossentropy', optimizer=tfko.Adam(lr=self.lr))
        return model

    def __init__(self, action_size, observation_size, training_param=TrainingParamSAC()):
        self.action_size = action_size
        self.observation_size = observation_size
        self.training_param = training_param

        # For optimizers
        self.lr = self.training_param.lr

        # Models
        self.model_Q = self._build_q_NN()
        self.model_Q2 = self._build_q_NN()

        self.model_value = self._build_value_NN()
        self.model_value_target = self._build_value_NN()
        self.model_value_target.set_weights(self.model_value.get_weights())

        self.model_policy = self._build_policy_NN()

        # For automatic alpha/temperature tuning.
        self._alpha = tf.Variable(training_param.ALPHA)

        # These are used in the get_eye functions
        self.previous_size = 0
        self.previous_eyes = None
        self.previous_arange = None
        self.previous_size_train = 0
        self.previous_eyes_train = None

        # Statistics
        self.average_reward = 0
        self.life_spent = 1
        self.Is_nan = False

        # Deques for calculating moving averages of losses
        self.Q_loss_30 = deque(maxlen=30)
        self.Q2_loss_30 = deque(maxlen=30)
        self.policy_loss_30 = deque(maxlen=30)
        self.value_loss_30 = deque(maxlen=30)
        self.alpha_loss_30 = deque(maxlen=30)


    def predict_movement_stochastic(self, data, batch_size=None):
        """ Stochastic policy """
        if batch_size is None:
            batch_size = data.shape[0]
        # Policy outputs a probability distribution over the actions
        p_actions = self.model_policy.predict(data, batch_size=batch_size)
        # create a distribution to sample from
        m = tfp.distributions.Categorical(probs=p_actions)
        # sample action from distribution
        action = m.sample()
        # Get probability for the chosen action
        prob = m.prob(action)
        return action, prob

    def predict_movement_evaluate(self, data, nr_acts):
        # Policy outputs a probability distribution over the actions
        p_actions = self.model_policy.predict(data, batch_size=1).squeeze()

        # Choose the nr_acts actions with the highest probabilities
        best_actions = np.argsort(p_actions)[::-1]
        best_actions = best_actions[:nr_acts]

        return best_actions, p_actions[best_actions]

    def predict_movement(self, data, batch_size=None, epsilon=0.0):
        """ Predict movement "deterministic"  """
        if batch_size is None:
            batch_size = data.shape[0]

        # Policy outputs a probability distribution over the actions
        p_actions = self.model_policy.predict(data, batch_size=batch_size)

        # Choose the action with the highest probability
        opt_policy_orig = np.argmax(np.abs(p_actions), axis=-1)

        opt_policy = 1.0 * opt_policy_orig

        # With epsilon probability, make actions random instead of using suggestion from policy network =======
        # TODO: choose random action with stochastic_predict_movement?
        rand_val = np.random.random(data.shape[0])
        opt_policy[rand_val < epsilon] = np.random.randint(0, self.action_size, size=(np.sum(rand_val < epsilon)))
        # =====================================================================================================

        opt_policy = opt_policy.astype(np.int)

        return opt_policy, p_actions[:, opt_policy]

    def get_eye_pm(self, batch_size):
        if batch_size != self.previous_size:
            tmp = np.zeros((batch_size, self.action_size), dtype=np.float32)
            self.previous_eyes = tmp
            self.previous_arange = np.arange(batch_size)
            self.previous_size = batch_size
        return self.previous_eyes, self.previous_arange

    def get_eye_train(self, batch_size):
        if batch_size != self.previous_size_train:
            self.previous_eyes_train = np.repeat(np.eye(self.action_size),
                                                 batch_size * np.ones(self.action_size, dtype=np.int),
                                                 axis=0)
            self.previous_eyes_train = tf.convert_to_tensor(self.previous_eyes_train, dtype=tf.float32)
            self.previous_size_train = batch_size
        return self.previous_eyes_train

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """Trains networks to fit given parameters"""
        if batch_size is None:
            batch_size = s_batch.shape[0]

        self.life_spent += 1  # increase counter
        self._alpha = 1 / np.log(self.life_spent) / 2

        # (1) training of the Q-FUNCTION networks ######################################################################
        # Calculate losses and do one optimizer step each for Q and Q2 networks
        Q_loss, Q2_loss = self._train_Q_networks(s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size)

        # (2) training of the POLICY network ###########################################################################
        # Calculate loss and do one optimizer step for policy network
        policy_loss = self._train_policy_network(s_batch, batch_size)

        # (3) training of value function
        value_loss = self._train_value_network(s_batch, a_batch, batch_size)

        # (4) save statistics to tensorboard logs
        if tf_writer is not None:
            self.Q_loss_30.append(Q_loss)
            self.Q2_loss_30.append(Q2_loss)
            self.policy_loss_30.append(policy_loss)
            self.value_loss_30.append(value_loss)

            if (self.life_spent-1) % 10 == 0:  # Change for less updates
                with tf_writer.as_default():
                    tf.summary.scalar("loss/Q1_loss_30", np.mean(self.Q_loss_30), self.life_spent)
                    tf.summary.scalar("loss/Q2_loss_30", np.mean(self.Q2_loss_30), self.life_spent)
                    tf.summary.scalar("loss/policy_loss_30", np.mean(self.policy_loss_30), self.life_spent)
                    tf.summary.scalar("loss/value_loss_30", np.mean(self.value_loss_30), self.life_spent)

                    tf.summary.scalar("alpha/alpha", self._alpha, self.life_spent)

        losses = (Q_loss, Q2_loss, policy_loss, value_loss)
        return np.isfinite(losses).all()

    def target_train(self):
        """ Update weights of target network """
        TAU = self.training_param.TAU

        model_weights = self.model_value.get_weights()
        target_model_weights = self.model_value_target.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.model_value_target.set_weights(target_model_weights)

    def _train_Q_networks(self, s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size):
        # (1) Find the current estimate of the next state value.
        next_state_value = self.model_value_target.predict(s2_batch, batch_size=batch_size).squeeze(-1)

        target = r_batch + (1 - d_batch) * self.training_param.DECAY_RATE * next_state_value

        # (3) Add information about which action was taken by setting last_action[batch_index, action(batch_index)] = 1
        last_action = np.zeros((batch_size, self.action_size))
        last_action[np.arange(batch_size), a_batch] = 1

        # (4) train on batch
        Q1_loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        Q2_loss = self.model_Q2.train_on_batch([s_batch, last_action], target)

        return Q1_loss, Q2_loss

    def _train_policy_network(self, s_batch, batch_size):
        # Create a huge matrix of shape (action_size*batch_size, action_size). It is NOT batch_size identity
        # matrices stacked on top of each other, but rather something like: [1,0,0] (batch size times), [0,1,0,...]
        # batch size time etc. Stored as a class/instance variable after it has been created the first time.
        eye_train = self.get_eye_train(batch_size)

        # Create a huge matrix of shape (batch_size*action_size, observation_size). It is essentially action_size copies
        # of s_batch stacked on top of each other.
        tiled_s_batch = np.tile(s_batch, (self.action_size, 1))
        tiled_s_batch_ts = tf.convert_to_tensor(tiled_s_batch)

        # Use the large tiled matrices above to do one big forward pass of the Q-networks. The result without reshaping
        # is an array of shape (batch_size*action_size, 1), where the first batch_size elements are the Q-values for
        # action a_0, etc. After reshaping, we get a matrix of shape (batch_size, action_size) filled with Q-values.
        action_values_Q1 = self.model_Q.predict([tiled_s_batch_ts, eye_train],
                                                batch_size=batch_size).reshape(batch_size, -1)
        action_values_Q2 = self.model_Q2.predict([tiled_s_batch_ts, eye_train],
                                                 batch_size=batch_size).reshape(batch_size, -1)
        action_values_min = np.fmin(action_values_Q1, action_values_Q2)

        advantage = action_values_min - np.amax(action_values_min, axis=-1).reshape(batch_size, 1)

        temp = self._alpha  # "temperature"

        # Calculate a probability distribution over the actions (one distribution for every sample in the batch).
        # Here the temperature parameter comes into play!
        new_proba = np.exp(advantage / temp) / np.sum(np.exp(advantage / temp), axis=-1).reshape(batch_size, 1)
        new_proba_ts = tf.convert_to_tensor(new_proba)

        # The loss function used is the categorical cross-ENTROPY loss = - sum_a (new_proba(a) * log(policy(s, a))
        policy_loss = self.model_policy.train_on_batch(s_batch, new_proba_ts)
        # =========================================================================

        return policy_loss

    def _train_value_network(self, s_batch, a_batch, batch_size):
        tiled_s_batch = np.tile(s_batch, (self.action_size, 1))
        tiled_s_batch_ts = tf.convert_to_tensor(tiled_s_batch)
        eye_train = self.get_eye_train(batch_size)

        action_values_Q1 = self.model_Q.predict([tiled_s_batch_ts, eye_train],
                                                batch_size=batch_size).reshape(batch_size, -1)
        action_values_Q2 = self.model_Q2.predict([tiled_s_batch_ts, eye_train],
                                                 batch_size=batch_size).reshape(batch_size, -1)

        value_target = np.fmin(action_values_Q1, action_values_Q2)

        value_target = value_target[np.arange(batch_size), a_batch]

        target_pi = self.model_policy.predict(s_batch, batch_size=batch_size)

        value_target = value_target - np.sum(target_pi * np.log(target_pi + 1e-6), axis=-1)

        loss_value = self.model_value.train_on_batch(s_batch, value_target)

        return loss_value

    @staticmethod
    def _get_path_model(path, name=None):
        path_Q = os.path.join(path, 'Q')
        path_Q2 = os.path.join(path, 'Q2')
        path_value = os.path.join(path, 'value')
        path_value_target = os.path.join(path, 'value_target')
        path_policy = os.path.join(path, 'policy')
        return path_Q, path_Q2, path_value, path_value_target, path_policy

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        path_Q, path_Q2, path_value, path_value_target, path_policy = self._get_path_model(path, name)
        self.model_Q.save('{}.{}'.format(path_Q, ext))
        self.model_Q2.save('{}.{}'.format(path_Q2, ext))
        self.model_value.save('{}.{}'.format(path_value, ext))
        self.model_value_target.save('{}.{}'.format(path_value_target, ext))
        self.model_policy.save('{}.{}'.format(path_policy, ext))
        print("Successfully saved network.")

    def load_network(self, path, name=None, ext="h5"):
        path_Q, path_Q2, path_value, path_value_target, path_policy = self._get_path_model(path,
                                                                                                               name)
        self.model_Q = load_model('{}.{}'.format(path_Q, ext))
        self.model_Q2 = load_model('{}.{}'.format(path_Q2, ext))
        self.model_value = load_model('{}.{}'.format(path_value, ext))
        self.model_value_target = load_model('{}.{}'.format(path_value_target, ext))
        self.model_policy = load_model('{}.{}'.format(path_policy, ext))
        print("Succesfully loaded network.")
