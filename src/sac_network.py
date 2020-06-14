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

from l2rpn_baselines.utils import BaseDeepQ  # Baseline Q network.
from sac_training_param import TrainingParamSAC


class SACNetwork(object):
    # TODO: Should all networks have the same optimizer learning rate/decay?
    # TODO: change trining function according to the non-stochastic policy (take argmax)...

    def make_optimiser(self, lr, lr_decay_steps, lr_decay_rate):
        schedule = tfko.schedules.InverseTimeDecay(lr, lr_decay_steps, lr_decay_rate)
        return schedule, tfko.Adam(learning_rate=schedule)

    def construct_q_network(self):
        """ Essentially copied (but cleaned up) from SAC_NN"""
        # Double Q networks
        self.model_Q = self._build_q_NN()
        self.model_Q2 = self._build_q_NN()

        # Compile
        self.model_Q.compile(loss='mse', optimizer=self.optimizer_Q)
        self.model_Q2.compile(loss='mse', optimizer=self.optimizer_Q2)

        # Double Q targets
        self.model_Q_target = self._build_q_NN()
        self.model_Q2_target = self._build_q_NN()

        # Target networks are never trained with optimizer, but compile anyway
        self.model_Q_target.compile(loss='mse', optimizer=self.optimizer_Q)
        self.model_Q2_target.compile(loss='mse', optimizer=self.optimizer_Q2)

        # policy function approximation
        self.model_policy = self._build_policy_NN()
        self.model_policy.compile(loss='categorical_crossentropy', optimizer=self.optimizer_policy)

        print("Successfully constructed networks.")

    def _build_q_NN(self):
        """ Build Q networks as in the baseline SAC """
        input_states = Input(shape=(self.observation_size))
        input_action = Input(shape=(self.action_size))
        input_layer = Concatenate()([input_states, input_action])

        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)

        # lay2 = Dense(self.observation_size)(lay1)
        # lay2 = Activation('relu')(lay2)

        lay3 = Dense(2 * self.action_size)(lay1)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(1, activation='linear')(lay3)

        model = Model(inputs=[input_states, input_action], outputs=[advantage])
        return model

    def _build_policy_NN(self):
        """ Build policy network as in the baseline SAC """
        input_states = Input(shape=(self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        # lay2 = Dense(self.observation_size)(lay1)
        # lay2 = Activation('relu')(lay2)

        lay3 = Dense(2 * self.action_size)(lay1)
        lay3 = Activation('relu')(lay3)

        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)
        model_policy = Model(inputs=[input_states], outputs=[soft_proba])
        return model_policy

    def __init__(self, action_size, observation_size, training_param=TrainingParamSAC()):
        self.action_size = action_size
        self.observation_size = observation_size
        self.training_param = training_param

        # For optimizers
        self.lr = self.training_param.lr
        self.lr_decay_steps = self.training_param.learning_rate_decay_steps
        self.lr_decay_rate = self.training_param.learning_rate_decay_rate

        # Optimizers
        self.schedule_lr_Q, self.optimizer_Q = \
            self.make_optimiser(self.lr, self.lr_decay_steps, self.lr_decay_rate)
        self.schedule_lr_Q2, self.optimizer_Q2 = \
            self.make_optimiser(self.lr, self.lr_decay_steps, self.lr_decay_rate)
        self.schedule_lr_policy, self.optimizer_policy = \
            self.make_optimiser(self.lr, self.lr_decay_steps, self.lr_decay_rate)

        # Models
        self.model_Q = None
        self.model_Q2 = None
        self.model_Q_target = None
        self.model_Q2_target = None
        self.model_policy = None

        # Define and compile the networks (with the optimizers above)
        self.construct_q_network()

        # For automatic alpha/temperature tuning.
        self._alpha = tf.Variable(training_param.ALPHA)

        self._automatic_alpha_tuning = training_param.AUTOMATIC_ALPHA_TUNING
        if self._automatic_alpha_tuning:
            self._log_alpha = tf.Variable(tf.math.log(self._alpha))

            self._alpha_lr = training_param.ALPHA_LR
            self._alpha_optimizer = tf.optimizers.Adam(self._alpha_lr, name='alpha_optimizer')
            # The paper: "SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS" https://arxiv.org/pdf/1910.07207.pdf
            # suggests to set the target entropy to 0.98 * (np.log(action_size)) which is very close to the maximum
            # entropy np.log(action_size) ??
            self._target_entropy = 0.98 * (np.log(action_size))  # TODO how to set this???

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

        self.life_spent += 1  # increase counter = number of optimizer steps + 1

        # (1) training of the Q-FUNCTION networks ######################################################################
        # Calculate losses and do one optimizer step each for Q and Q2 networks
        Q_loss, Q2_loss = self._train_Q_networks(s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size)

        # (2) training of the POLICY network ###########################################################################
        # Calculate loss and do one optimizer step for policy network
        policy_loss = self._train_policy_network(s_batch, batch_size)

        # (3) tune alpha/temperature parameter #########################################################################
        if self._automatic_alpha_tuning:
            # Calculate loss and do one optimizer step for alpha
            alpha_loss = self._adjust_alpha(a_batch, s_batch, batch_size)
        else:
            self._alpha = 1 / np.log(self.life_spent) / 2  # TODO: Keep alpha if not stochastic policy?
            alpha_loss = -1

        # (4) save statistics to tensorboard logs
        if tf_writer is not None:
            self.Q_loss_30.append(Q_loss)
            self.Q2_loss_30.append(Q2_loss)
            self.policy_loss_30.append(policy_loss)
            self.alpha_loss_30.append(alpha_loss)

            if self.life_spent % 1 == 0:  # Change for less updates
                with tf_writer.as_default():
                    tf.summary.scalar("loss/Q1_loss_30", np.mean(self.Q_loss_30), self.life_spent)
                    tf.summary.scalar("loss/Q2_loss_30", np.mean(self.Q2_loss_30), self.life_spent)
                    tf.summary.scalar("loss/policy_loss_30", np.mean(self.policy_loss_30), self.life_spent)
                    tf.summary.scalar("alpha/alpha", self._alpha, self.life_spent)
                    if self._automatic_alpha_tuning:
                        tf.summary.scalar("alpha/alpha_loss_30", np.mean(self.alpha_loss_30), self.life_spent)

        losses = (Q_loss, Q2_loss, policy_loss, alpha_loss)
        return np.isfinite(losses).all()

    def target_train(self):
        """ Update weights of target Q networks """
        tau = self.training_param.TAU

        # Update Q target
        Q_weights = self.model_Q.get_weights().copy()
        Q_target_weights = self.model_Q_target.get_weights().copy()
        for i in range(len(Q_weights)):
            Q_target_weights[i] = tau * Q_weights[i] + (1 - tau) * Q_target_weights[i]
        self.model_Q_target.set_weights(Q_target_weights)

        # Update Q2 target
        Q2_weights = self.model_Q2.get_weights().copy()
        Q2_target_weights = self.model_Q2_target.get_weights()
        for i in range(len(Q2_weights)):
            Q2_target_weights[i] = tau * Q2_weights[i] + (1 - tau) * Q2_target_weights[i]
        self.model_Q2_target.set_weights(Q2_target_weights)

    def _train_Q_networks(self, s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size):
        # (1) Find the current estimate of the next state value.
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

        # Estimate value of the next state
        # According to SAC Discrete paper =============================
        # next_action_values = target_pi * (next_action_values - self._alpha * np.log(target_pi + 1e-6))
        # next_state_value = np.sum(next_action_values, axis=-1)  # Sum over the actions (not over the batch)
        # =============================================================

        # ALTERNATIVE: Use the choice of action that is used at evaluation ======
        next_action = np.argmax(target_pi, axis=-1)
        next_state_value = next_action_values[np.arange(batch_size), next_action]
        # =======================================================================

        # (2) Bellman. The "target" for the Q networks is the expected reward = sum of immediate reward r_batch and the
        # discounted value of the next state (predicted by the target Q networks)
        target = np.zeros((batch_size, 1))
        target[:, 0] = r_batch + (1 - d_batch) * self.training_param.DECAY_RATE * next_state_value

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

        # Attempt to implement according to the papers: ===============================================
        # policy loss = E [π(s)T [α log(π(s)) − Q(s)]] with expectation over data
        # target_pi = self.model_policy.predict(s_batch, batch_size=batch_size)
        #
        # with tf.GradientTape() as tape:
        #     pi = self.model_policy(s_batch)  # shape (batch_size, action_size)
        #     log_pi = tf.math.log(pi + 1e-6)  # get log probabilities
        #     policy_loss = self._alpha * log_pi - action_values_min
        #     # Do the scalar product for all samples in the batch
        #     policy_loss = target_pi * policy_loss
        #     policy_loss = tf.reduce_sum(policy_loss, axis=-1)
        #     # The loss is the expected loss over the data, so take the mean over the batch as estimate
        #     policy_loss = tf.reduce_mean(policy_loss)
        #
        # # Get the gradient and take an optimizer step
        # grads = tape.gradient(policy_loss, self.model_policy.trainable_variables)
        # self.optimizer_policy.apply_gradients(zip(grads, self.model_policy.trainable_variables))
        # ===============================================================================================

        # ALTERNATIVE: For "deterministic" policy =============================
        # Calculate the "advantage" of all actions as compared to the "optimal" (greedy) action.
        # Advantage = Q(s,a) - V(s) (I have renamed action_v1 --> advantage). All advantage values are <= 0.

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

    def _adjust_alpha(self, a_batch, s_batch, batch_size):
        """ TODO: this is not working correctly """
        if not isinstance(self._target_entropy, Number):
            self._target_entropy = 0.0

        action_probabilities = self.model_policy.predict(s_batch, batch_size=batch_size)
        log_pis = np.log(action_probabilities + 1e-6)

        current_entropy = -1.0 * tf.math.reduce_sum(action_probabilities * log_pis, axis=-1)
        current_entropy = tf.math.reduce_mean(current_entropy)
        entropy_diff = current_entropy - self._target_entropy
        with tf.GradientTape() as tape:
            alpha_loss = self._log_alpha*tf.math.reduce_mean(entropy_diff)
            #alpha_losses = -1.0 * (self._log_alpha * tf.stop_gradient(log_pis + self._target_entropy))
            # Take the expectation over the actions
            #alpha_losses = tf.math.reduce_sum(action_probabilities * alpha_losses, axis=-1)
            #alpha_loss = tf.math.reduce_mean(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._log_alpha]))
        self._alpha = tf.math.exp(self._log_alpha)

        return alpha_loss

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
        path_Q, path_Q2, path_Q_target, path_Q2_target, path_policy = self._get_path_model(path, name)
        self.model_Q.save('{}.{}'.format(path_Q, ext))
        self.model_Q2.save('{}.{}'.format(path_Q2, ext))
        self.model_Q_target.save('{}.{}'.format(path_Q_target, ext))
        self.model_Q2_target.save('{}.{}'.format(path_Q2_target, ext))
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
