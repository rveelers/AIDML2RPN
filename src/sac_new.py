"""Define SACNetwork used by SACAgent."""
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque

import warnings  # tf2.0 friendly
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input, Concatenate
    import tensorflow.keras.optimizers as tfko
    from tensorflow.keras.models import load_model

from sac_training_param import TrainingParamSAC


class SACNetwork(object):
    """ Class containing the neural networks used by the SACAgent. """

    def _build_q_NN(self):
        """Build and compile a Q network."""
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
        model.compile(loss='mse', optimizer=tfko.Adam(lr=self.training_param.lr))
        return model

    def _build_value_NN(self):
        """Build and compile a value network."""
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
        model.compile(loss='mse', optimizer=tfko.Adam(lr=self.training_param.lr))
        return model

    def _build_policy_NN(self):
        """Build and compile a policy network."""
        input_states = Input(shape=(self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(self.action_size*2)(lay2)
        lay3 = Activation('relu')(lay3)

        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)

        model = Model(inputs=[input_states], outputs=[soft_proba])
        model.compile(loss='categorical_crossentropy', optimizer=tfko.Adam(lr=self.training_param.lr))
        return model

    def __init__(self, action_size, observation_size, training_param=TrainingParamSAC()):
        self.action_size = action_size
        self.observation_size = observation_size

        # Class with training parameters such as learning rate
        self.training_param = training_param

        # Build and compile models
        self.model_Q = self._build_q_NN()
        self.model_Q2 = self._build_q_NN()

        self.model_value = self._build_value_NN()
        self.model_value_target = self._build_value_NN()
        self.model_value_target.set_weights(self.model_value.get_weights())

        self.model_policy = self._build_policy_NN()

        # Temperature parameter. This is set manually in the training loop.
        self.alpha = None

        # These are used in the get_eye_train function which is useful for predicting on a batch.
        # By storing the
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

    def predict_movement(self, s_batch, batch_size=None, epsilon=0.0):
        """Epsilon-greedy policy. Return an action based on the policy with epsilon probability, else random.

        Works on a batch of data.
        """
        if batch_size is None:
            batch_size = s_batch.shape[0]

        # Policy outputs a probability distribution over the actions
        p_actions = self.model_policy.predict(s_batch, batch_size=batch_size)

        # Choose the action with the highest probability
        opt_policy = 1.0 * np.argmax(np.abs(p_actions), axis=-1)

        # With epsilon probability, make actions random instead of using suggestion from policy network =======
        rand_val = np.random.random(s_batch.shape[0])
        opt_policy[rand_val < epsilon] = np.random.randint(0, self.action_size, size=(np.sum(rand_val < epsilon)))
        # =====================================================================================================

        opt_policy = opt_policy.astype(np.int)

        return opt_policy, p_actions[:, opt_policy]

    def predict_movement_stochastic(self, s_batch, batch_size=None):
        """Stochastic policy. Return actions by sampling from the policy distribution.

        Works on a batch of data.
        """
        if batch_size is None:
            batch_size = s_batch.shape[0]

        # Policy outputs a probability distribution over the actions
        p_actions = self.model_policy.predict(s_batch, batch_size=batch_size)

        # create a distribution to sample from
        m = tfp.distributions.Categorical(probs=p_actions)

        # sample action from distribution
        action = m.sample()

        # Get probability for the chosen action
        prob = m.prob(action)

        return action, prob

    def predict_movement_evaluate(self, transformed_observation, nr_acts=1):
        """Policy for evaluation. Get the nr_acts top actions suggested by the policy network.

        This function only works with a batch size of one.
        """
        # Policy outputs a probability distribution over the actions.
        p_actions = self.model_policy.predict(transformed_observation, batch_size=1).squeeze()

        # Choose the nr_acts actions with the highest probabilities.
        best_actions = np.argsort(p_actions)[::-1]
        best_actions = best_actions[:nr_acts]

        # Return the best actions and their associated probabilities.
        return best_actions, p_actions[best_actions]

    def get_eye_train(self, batch_size):
        """Create a matrix with a specific structure which is useful for batch prediction.

        The generated matrix is of shape (action_size*batch_size, action_size) with the structure:
        [[1,0,0,...] (batch size times), [0,1,0,...] (batch size times), ...].
        This is useful for predicting on batches when we want to predict for ALL actions.
        """
        if batch_size != self.previous_size_train:
            self.previous_eyes_train = np.repeat(np.eye(self.action_size),
                                                 batch_size * np.ones(self.action_size, dtype=np.int),
                                                 axis=0)
            self.previous_eyes_train = tf.convert_to_tensor(self.previous_eyes_train, dtype=tf.float32)
            self.previous_size_train = batch_size
        return self.previous_eyes_train

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """Train networks to fit given parameters. Return True iff all losses are finite.

        s_batch: batch of states s.
        a_batch: batch of actions a.
        r_batch: batch of rewards r.
        d_batch: batch of done-flags d. d == 1 of game over state is reached, otherwise d == 0.
        s2_batch: batch of next states s2.
        tf_writer: tensorboard writer used for logging training statistics.
        """
        if batch_size is None:
            batch_size = s_batch.shape[0]

        self.life_spent += 1  # increase counter
        self.alpha = 1 / np.log(self.life_spent) / 2  # Manually set alpha

        # (1) training of the Q-FUNCTION networks
        # Calculate losses and do one optimizer step each for Q and Q2 networks
        Q_loss, Q2_loss = self._train_Q_networks(s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size)

        # (2) training of the POLICY network
        # Calculate loss and do one optimizer step for policy network
        policy_loss = self._train_policy_network(s_batch, batch_size)

        # (3) training of VALUE function
        value_loss = self._train_value_network(s_batch, batch_size)

        # (4) save statistics to tensorboard logs
        if tf_writer is not None:
            self.Q_loss_30.append(Q_loss)
            self.Q2_loss_30.append(Q2_loss)
            self.policy_loss_30.append(policy_loss)
            self.value_loss_30.append(value_loss)

            if (self.life_spent-1) % 10 == 0:  # Change this to write statistics less often.
                with tf_writer.as_default():
                    tf.summary.scalar("loss/Q1_loss_30", np.mean(self.Q_loss_30), self.life_spent)
                    tf.summary.scalar("loss/Q2_loss_30", np.mean(self.Q2_loss_30), self.life_spent)
                    tf.summary.scalar("loss/policy_loss_30", np.mean(self.policy_loss_30), self.life_spent)
                    tf.summary.scalar("loss/value_loss_30", np.mean(self.value_loss_30), self.life_spent)

                    tf.summary.scalar("alpha/alpha", self.alpha, self.life_spent)

        losses = (Q_loss, Q2_loss, policy_loss, value_loss)
        return np.isfinite(losses).all()

    def target_train(self):
        """Update weights of target value network."""
        TAU = self.training_param.TAU

        # Get model/target weights and calculate their element-wise weighted average
        model_weights = self.model_value.get_weights()
        target_model_weights = self.model_value_target.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]

        # Set weights of the target network to the new weights
        self.model_value_target.set_weights(target_model_weights)

    def _train_Q_networks(self, s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size):
        """Train the Q networks on a batch of data. Return the loss values."""
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
        """Train the policy network on a batch of data. Return the loss value."""
        # Get a matrix with a specific structure which is useful for the batch prediction.
        # The generated matrix is of shape (action_size*batch_size, action_size) with the structure:
        #     [[1,0,0,...] (batch size times), [0,1,0,...] (batch size times), ...].
        eye_train = self.get_eye_train(batch_size)

        # Create a huge matrix of shape (batch_size*action_size, observation_size). It is essentially action_size copies
        # of s_batch stacked on top of each other.
        tiled_s_batch = np.tile(s_batch, (self.action_size, 1))
        tiled_s_batch_ts = tf.convert_to_tensor(tiled_s_batch)

        # Use the large tiled matrices above to do one big forward pass of the Q-networks. The result without reshaping
        # is an array of shape (batch_size*action_size, 1), where the first batch_size elements are the Q-values for
        # action a_0, etc. After reshaping, we get a matrix of shape (batch_size, action_size) filled with Q-values.
        action_values_Q1 = self.model_Q.predict([tiled_s_batch_ts, eye_train],
                                                batch_size=batch_size).reshape(-1, batch_size).T
        action_values_Q2 = self.model_Q2.predict([tiled_s_batch_ts, eye_train],
                                                 batch_size=batch_size).reshape(-1, batch_size).T
        # Take the element-wise min
        action_values_min = np.fmin(action_values_Q1, action_values_Q2)

        advantage = action_values_min - np.amax(action_values_min, axis=-1).reshape(batch_size, 1)

        temp = self.alpha  # "temperature"

        # Calculate a probability distribution over the actions (one distribution for every sample in the batch).
        # Here the temperature parameter comes into play!
        new_proba = np.exp(advantage / temp) / np.sum(np.exp(advantage / temp), axis=-1).reshape(batch_size, 1)
        new_proba_ts = tf.convert_to_tensor(new_proba)

        # The loss function used is the categorical cross-entropy loss: - sum_a (new_proba(a) * log(policy(s, a))
        policy_loss = self.model_policy.train_on_batch(s_batch, new_proba_ts)
        # =========================================================================

        return policy_loss

    def _train_value_network(self, s_batch, batch_size):
        """Train the value network on a batch of data. Return the loss value."""
        # Create a huge matrix of shape (batch_size*action_size, observation_size). It is essentially action_size copies
        # of s_batch stacked on top of each other.
        tiled_s_batch = np.tile(s_batch, (self.action_size, 1))
        tiled_s_batch_ts = tf.convert_to_tensor(tiled_s_batch)

        # Get a matrix with a specific structure which is useful for the batch prediction.
        # The generated matrix is of shape (action_size*batch_size, action_size) with the structure:
        #     [[1,0,0,...] (batch size times), [0,1,0,...] (batch size times), ...].
        eye_train = self.get_eye_train(batch_size)

        action_values_Q1 = self.model_Q.predict([tiled_s_batch_ts, eye_train],
                                                batch_size=batch_size).reshape(-1, batch_size).T
        action_values_Q2 = self.model_Q2.predict([tiled_s_batch_ts, eye_train],
                                                 batch_size=batch_size).reshape(-1, batch_size).T
        # Take the element-wise minimum
        action_values = np.fmin(action_values_Q1, action_values_Q2)

        # Get the actions according to the policy
        target_pi = self.model_policy.predict(s_batch, batch_size=batch_size)
        actions = np.argmax(target_pi, axis=-1)

        # Calculate the target values
        value_target = action_values[np.arange(batch_size), actions]
        value_target = value_target - np.sum(target_pi * np.log(target_pi + 1e-6), axis=-1)

        # Train on batch
        loss_value = self.model_value.train_on_batch(s_batch, value_target)

        return loss_value

    @staticmethod
    def _get_path_model(path, name=None):
        """Return paths for the five networks."""
        path_Q = os.path.join(path, 'Q')
        path_Q2 = os.path.join(path, 'Q2')
        path_value = os.path.join(path, 'value')
        path_value_target = os.path.join(path, 'value_target')
        path_policy = os.path.join(path, 'policy')
        return path_Q, path_Q2, path_value, path_value_target, path_policy

    def save_network(self, path, name=None, ext="h5"):
        """Save the neural networks at the specified path."""
        path_Q, path_Q2, path_value, path_value_target, path_policy = self._get_path_model(path, name)
        self.model_Q.save('{}.{}'.format(path_Q, ext))
        self.model_Q2.save('{}.{}'.format(path_Q2, ext))
        self.model_value.save('{}.{}'.format(path_value, ext))
        self.model_value_target.save('{}.{}'.format(path_value_target, ext))
        self.model_policy.save('{}.{}'.format(path_policy, ext))
        print("Successfully saved network.")

    def load_network(self, path, name=None, ext="h5"):
        """Load the neural networks from the specified path."""
        path_Q, path_Q2, path_value, path_value_target, path_policy = self._get_path_model(path, name)
        self.model_Q = load_model('{}.{}'.format(path_Q, ext))
        self.model_Q2 = load_model('{}.{}'.format(path_Q2, ext))
        self.model_value = load_model('{}.{}'.format(path_value, ext))
        self.model_value_target = load_model('{}.{}'.format(path_value_target, ext))
        self.model_policy = load_model('{}.{}'.format(path_policy, ext))
        print("Succesfully loaded network.")
