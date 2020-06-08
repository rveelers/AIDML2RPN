# tf2.0 friendly
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numbers import Number

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input, Concatenate

from l2rpn_baselines.utils import BaseDeepQ
from sac_training_param import TrainingParamSAC

from collections import deque

class SACNetworkDiscrete(BaseDeepQ):

    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 learning_rate_decay_steps=1000,
                 learning_rate_decay_rate=0.95,
                 training_param=TrainingParamSAC()):
        BaseDeepQ.__init__(self, action_size, observation_size,
                           lr, learning_rate_decay_steps, learning_rate_decay_rate,
                           training_param)

        self.average_reward = 0
        self.life_spent = 1
        self.qvalue_evolution = np.zeros((0,))
        self.Is_nan = False

        self.model_Q = None
        self.model_Q2 = None
        self.model_Q_target = None
        self.model_Q2_target = None
        self.model_policy = None

        self.schedule_lr_Q, self.optimizer_Q = self.make_optimiser()
        self.schedule_lr_Q2, self.optimizer_Q2 = self.make_optimiser()
        self.schedule_lr_policy, self.optimizer_policy = self.make_optimiser()

        self.construct_q_network()

        self.previous_size = 0
        self.previous_eyes = None
        self.previous_arange = None
        self.previous_size_train = 0
        self.previous_eyes_train = None

        # For automatic alpha/temperature tuning.
        self._alpha = tf.Variable(0.05)
        self._automatic_alpha_tuning = training_param.AUTOMATIC_ALPHA_TUNING
        if self._automatic_alpha_tuning:
            self._alpha_lr = training_param.ALPHA_LR
            self._alpha_optimizer = tf.optimizers.Adam(self._alpha_lr, name='alpha_optimizer')
            # Set the target entropy according to the paper: "SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS"
            # https://arxiv.org/pdf/1910.07207.pdf
            self._target_entropy = 0.98 * (np.log(action_size))

        # Deques for calculating moving averages of losses
        self.Q_loss_30 = deque(maxlen=30)
        self.Q2_loss_30 = deque(maxlen=30)
        self.policy_loss_30 = deque(maxlen=30)
        self.alpha_loss_30 = deque(maxlen=30)


    def _build_q_NN(self):
        input_states = Input(shape=(self.observation_size))
        input_action = Input(shape=(self.action_size))
        input_layer = Concatenate()([input_states, input_action])

        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(2 * self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(1, activation='linear')(lay3)

        model = Model(inputs=[input_states, input_action], outputs=[advantage])
        return model

    def _build_policy_NN(self):
        # proba of choosing action a depending on policy pi
        input_states = Input(shape=(self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(2 * self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)

        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)
        model_policy = Model(inputs=[input_states], outputs=[soft_proba])
        return model_policy

    def construct_q_network(self):
        # construct double Q networks
        self.model_Q = self._build_q_NN()
        self.model_Q2 = self._build_q_NN()

        # Compile
        self.model_Q.compile(loss='mse', optimizer=self.optimizer_Q)
        self.model_Q2.compile(loss='mse', optimizer=self.optimizer_Q2)

        # Double Q targets
        self.model_Q_target = self._build_q_NN()
        self.model_Q2_target = self._build_q_NN()

        # Is never trained, but compile anyway
        self.model_Q_target.compile(loss='mse', optimizer=self.optimizer_Q)
        self.model_Q2_target.compile(loss='mse', optimizer=self.optimizer_Q2)

        # policy function approximation
        self.model_policy = self._build_policy_NN()
        self.model_policy.compile(loss='categorical_crossentropy', optimizer=self.optimizer_policy)
        print("Successfully constructed networks.")

    def get_eye_pm(self, batch_size):
        if batch_size != self.previous_size:
            tmp = np.zeros((batch_size, self.action_size), dtype=np.float32)
            self.previous_eyes = tmp
            self.previous_arange = np.arange(batch_size)
            self.previous_size = batch_size
        return self.previous_eyes, self.previous_arange

    def predict_movement(self, data, epsilon, batch_size=None):
        """ Change (1): Deterministic --> stochastic policy """
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

    def get_eye_train(self, batch_size):
        if batch_size != self.previous_size_train:
            self.previous_eyes_train = np.repeat(np.eye(self.action_size),
                                                 batch_size * np.ones(self.action_size, dtype=np.int),
                                                 axis=0)
            self.previous_eyes_train = tf.convert_to_tensor(self.previous_eyes_train, dtype=tf.float32)
            self.previous_size_train = batch_size
        return self.previous_eyes_train

    def _train_Q_networks(self, s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size):
        # Create a huge matrix of shape (batch_size*action_size, observation_size). It is essentially action_size copies
        # of s_batch stacked on top of each other.
        tiled_s2_batch = np.tile(s2_batch, (self.action_size, 1))
        tiled_s2_batch_ts = tf.convert_to_tensor(tiled_s2_batch)

        # Create a huge matrix of shape (action_size*batch_size, action_size). It is NOT batch_size identity
        # matrices stacked on top of each other, but rather something like: [1,0,0] (batch size times), [0,1,0,...]
        # batch size time etc. Stored as a class/instance variable after it has been created the first time.
        eye_train = self.get_eye_train(batch_size)

        # Use the large tiled matrices above to do one big forward pass of the Q_target-networks. The result without
        # reshaping is an array of shape (batch_size*action_size, 1), where the first batch_size elements are the
        # Q-values for action a_0, etc. After reshaping, we get a matrix of shape (batch_size, action_size) filled
        # with Q-values.
        next_action_values_Q1 = self.model_Q_target.predict([tiled_s2_batch_ts, eye_train],
                                                            batch_size=batch_size).reshape(batch_size, -1)
        next_action_values_Q2 = self.model_Q2_target.predict([tiled_s2_batch_ts, eye_train],
                                                             batch_size=batch_size).reshape(batch_size, -1)
        # Take element-wise minimum
        next_action_values = np.fmin(next_action_values_Q1, next_action_values_Q2)

        target_pi = self.model_policy.predict(s2_batch, batch_size=batch_size)

        next_action_values = target_pi * (next_action_values - self._alpha * np.log(target_pi + 1e-6))
        next_state_value = np.sum(next_action_values, axis=-1)  # Sum over the actions (not over the batch)

        # Add information about which action was taken by setting last_action[batch_index, action(batch_index)] = 1
        last_action = np.zeros((batch_size, self.action_size))
        last_action[np.arange(batch_size), a_batch] = 1

        # Bellman. The "target" for the Q networks is the expected reward = sum of immediate reward r_batch and the
        # discounted value of the next state (predicted by the model_value_target network)
        target = np.zeros((batch_size, 1))
        target[:, 0] = r_batch + (1 - d_batch) * self.training_param.DECAY_RATE * next_state_value

        Q1_loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        Q2_loss = self.model_Q2.train_on_batch([s_batch, last_action], target)

        return Q1_loss, Q2_loss  # loss, loss_2

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

        target_pi = self.model_policy.predict(s_batch, batch_size=batch_size)

        with tf.GradientTape() as tape:
            pi = self.model_policy(s_batch)
            log_pi = tf.math.log(pi + 1e-6)
            policy_loss = self._alpha * log_pi - action_values_min
            policy_loss = target_pi * policy_loss
            policy_loss = tf.reduce_sum(policy_loss, axis=-1)
            policy_loss = tf.reduce_mean(policy_loss)
        grads = tape.gradient(policy_loss, self.model_policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads, self.model_policy.trainable_variables))

        return policy_loss

    def _adjust_alpha(self, a_batch, s2_batch, batch_size):
        if not isinstance(self._target_entropy, Number):
            self._target_entropy = 0.0

        action_probabilities = self.model_policy.predict(s2_batch, batch_size=batch_size)
        log_pis = np.log(action_probabilities[np.arange(batch_size), a_batch] + 1e-6)

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (self._alpha * log_pis) + self._target_entropy
            alpha_loss = tf.math.reduce_mean(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._alpha])
        self._alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._alpha]))

        return alpha_loss

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """Trains networks to fit given parameters"""
        if batch_size is None:
            batch_size = s_batch.shape[0]

        self.life_spent += 1  # increase counter

        # (1) training of the Q-FUNCTION networks ######################################################################
        Q_loss, Q2_loss = self._train_Q_networks(s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size)

        # (2) training of the POLICY network ###########################################################################
        policy_loss = self._train_policy_network(s_batch, batch_size)

        # (3) tune alpha/temperature parameter #########################################################################
        if self._automatic_alpha_tuning:
            alpha_loss = self._adjust_alpha(a_batch, s2_batch, batch_size)
        else:
            self._alpha = 1 / np.log(self.life_spent) / 2
            alpha_loss = -1

        # (4) save fun stats to tensorboard
        if tf_writer is not None:
            self.Q_loss_30.append(Q_loss)
            self.Q2_loss_30.append(Q2_loss)
            self.policy_loss_30.append(policy_loss)
            self.alpha_loss_30.append(alpha_loss)

            if self.life_spent % 1 == 0:  # Change for less updates
                with tf_writer.as_default():
                    tf.summary.scalar("alpha/alpha", self._alpha, self.life_spent)
                    tf.summary.scalar("loss/Q1_loss_30", np.mean(self.Q_loss_30), self.life_spent)
                    tf.summary.scalar("loss/Q2_loss_30", np.mean(self.Q2_loss_30), self.life_spent)
                    tf.summary.scalar("loss/policy_loss_30", np.mean(self.policy_loss_30), self.life_spent)
                    tf.summary.scalar("loss/alpha_loss_30", np.mean(self.alpha_loss_30), self.life_spent)

        losses = (Q_loss, Q2_loss, policy_loss, alpha_loss)
        return np.isfinite(losses).all()

    @staticmethod
    def _get_path_model(path, name=None):
        path_modelQ = os.path.join(path, 'Q')
        path_modelQ2 = os.path.join(path, 'Q2')
        path_modelQ_target = os.path.join(path, 'Q_target')
        path_modelQ2_target = os.path.join(path, 'Q2_target')
        path_policy = os.path.join(path, 'policy')
        return path_modelQ, path_modelQ2, path_modelQ_target, path_modelQ2_target, path_policy

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        path_modelQ, path_modelQ2, path_modelQ_target, path_modelQ2_target, path_policy = self._get_path_model(path,
                                                                                                               name)
        self.model_Q.save('{}.{}'.format(path_modelQ, ext))
        self.model_Q2.save('{}.{}'.format(path_modelQ2, ext))
        self.model_Q_target.save('{}.{}'.format(path_modelQ_target, ext))
        self.model_Q2_target.save('{}.{}'.format(path_modelQ2_target, ext))
        self.model_policy.save('{}.{}'.format(path_policy, ext))
        print("Successfully saved network.")

    def load_network(self, path, name=None, ext="h5"):
        path_modelQ, path_modelQ2, path_modelQ_target, path_modelQ2_target, path_policy = self._get_path_model(path,
                                                                                                               name)
        self.model_Q = load_model('{}.{}'.format(path_modelQ, ext))
        self.model_Q2 = load_model('{}.{}'.format(path_modelQ2, ext))
        self.model_Q_target = load_model('{}.{}'.format(path_modelQ_target, ext))
        self.model_Q2_target = load_model('{}.{}'.format(path_modelQ2_target, ext))
        self.model_policy = load_model('{}.{}'.format(path_policy, ext))
        print("Succesfully loaded network.")

    def target_train(self):
        Q_weights = self.model_Q.get_weights()
        Q_target_weights = self.model_Q_target.get_weights()
        for i in range(len(Q_weights)):
            Q_target_weights[i] = self.training_param.TAU * Q_weights[i] + (1 - self.training_param.TAU) * \
                                  Q_target_weights[i]
        self.model_Q_target.set_weights(Q_target_weights)

        Q2_weights = self.model_Q2.get_weights()
        Q2_target_weights = self.model_Q2_target.get_weights()
        for i in range(len(Q2_weights)):
            Q2_target_weights[i] = self.training_param.TAU * Q2_weights[i] + (1 - self.training_param.TAU) * \
                                   Q2_target_weights[i]
        self.model_Q2_target.set_weights(Q2_target_weights)
