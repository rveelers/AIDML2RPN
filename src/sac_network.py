import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from l2rpn_baselines.SAC.SAC_NN import SAC_NN
# from l2rpn_baselines.utils import TrainingParam

from sac_training_param import TrainingParamSAC


# TODO: how to pass training_params to this network?
# I want to use my "own" class TrainingParamSAC from sac_training_param. However, TrainingParam from
# l2rpn_baselines.utils seems to be used even though I pass TrainingParamSAC to the super()__init__() ??

# Specifically, when I call my_agent.train(...) on the SACAgent object (my_agent), the train function from
# DeepQAgent is called. On line 247 in DeepAgent, there is a check:
#      if training_step > max(training_param.MIN_OBSERVATION, training_param.MINIBATCH_SIZE):
# Since MIN_OBSERVATION is set to 5000 in the baselines, the training loop is not entered until step 5000 ...


class SACNetwork(SAC_NN):

    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 learning_rate_decay_steps=1000,
                 learning_rate_decay_rate=0.95,
                 training_param=TrainingParamSAC()):

        super().__init__(action_size,
                         observation_size,
                         lr=lr,
                         learning_rate_decay_steps=learning_rate_decay_steps,
                         learning_rate_decay_rate=learning_rate_decay_rate,
                         training_param=training_param)

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

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """Trains networks to fit given parameters"""
        if batch_size is None:
            batch_size = s_batch.shape[0]
        
        # (1) training of the Q-function networks ######################################################################
        # Save the graph just the first time
        if tf_writer is not None:
            tf.summary.trace_on()
        next_state_value = self.model_value_target.predict(s2_batch, batch_size=batch_size).reshape(-1)
        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.trace_export("model_value_target-graph", 0)
            tf.summary.trace_off()

        last_action = np.zeros((batch_size, self.action_size))
        # NEW: (Johan 2020-06-06)
        # Add information about which action was taken by setting last_action[batch_index, action(batch_index)] = 1
        last_action[np.arange(batch_size), a_batch] = 1
        # END NEW

        # Bellman. The "target" for the Q networks is the expected reward = sum of immediate reward r_batch and the
        # discounted value of the next state (predicted by the model_value_target network)
        target = np.zeros((batch_size, 1))
        target[:, 0] = r_batch + (1 - d_batch) * self.training_param.DECAY_RATE * next_state_value

        loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        loss_2 = self.model_Q2.train_on_batch([s_batch, last_action], target)

        # (2) training of the policy network ###########################################################################

        # Create a huge matrix of shape (batch_size*action_size, observation_size). It is essentially action_size copies
        # of s_batch stacked on top of each other.
        tiled_s_batch = np.tile(s_batch, (self.action_size, 1))
        tiled_s_batch_ts = tf.convert_to_tensor(tiled_s_batch)

        # Create a huge matrix of shape (action_size*batch_size, action_size). It is NOT batch_size identity
        # matrices stacked on top of each other, but rather something like: [1,0,0] (batch size times), [0,1,0,...]
        # batch size time etc. Stored as a class/instance variable after it has been created the first time.
        eye_train = self.get_eye_train(batch_size)

        # Use the large tiled matrices above to do one big forward pass of the Q-networks. The result without reshaping
        # is an array of shape (batch_size*action_size, 1), where the first batch_size elements are the Q-values for
        # action a_0, etc. After reshaping, we get a matrix of shape (batch_size, action_size) filled with Q-values.
        action_v1_orig = self.model_Q.predict([tiled_s_batch_ts, eye_train],
                                              batch_size=batch_size).reshape(batch_size, -1)
        action_v2_orig = self.model_Q2.predict([tiled_s_batch_ts, eye_train],
                                               batch_size=batch_size).reshape(batch_size, -1)

        # OLD:
        # action_v1 = action_v1_orig - np.amax(action_v1_orig, axis=-1).reshape(batch_size, 1)
        # NEW: (Johan 2020-06-06). Do the min{Q1, Q2} as specified in the paper?
        action_v_min = np.fmin(action_v1_orig, action_v2_orig)
        # END NEW

        # Calculate the "advantage" of all actions as compared to the optimal (greedy) action. Advantage = Q(s,a) - V(s)
        # (I have renamed action_v1 --> advantage). All advantage values are <= 0.
        advantage = action_v_min - np.amax(action_v_min, axis=-1).reshape(batch_size, 1)

        # Set the temperature parameter
        self.life_spent += 1
        temp = 1 / np.log(self.life_spent) / 2

        # Calculate a probability distribution over the actions (one distribution for every sample in the batch).
        # Here the temperature parameter comes into play!
        new_proba = np.exp(advantage / temp) / np.sum(np.exp(advantage / temp), axis=-1).reshape(batch_size, 1)
        new_proba_ts = tf.convert_to_tensor(new_proba)

        # The loss function used is the categorical cross-ENTROPY loss = - sum_a (new_proba(a) * log(policy(s, a))
        loss_policy = self.model_policy.train_on_batch(s_batch, new_proba_ts)

        # (3) training of the value function network ###################################################################
        target_pi = self.model_policy.predict(s_batch, batch_size=batch_size)

        # OLD:
        # value_target = np.fmin(action_v1_orig[0, a_batch], action_v2_orig[0, a_batch]) - np.sum(
        #      target_pi * np.log(target_pi + 1e-6))
        # NEW: (Johan 2020-06-06)
        # With the above implementation, only Q-values for the FIRST state in the batch is used.
        action_values_Q1 = action_v1_orig[np.arange(batch_size), a_batch]
        action_values_Q2 = action_v2_orig[np.arange(batch_size), a_batch]
        value_target = np.fmin(action_values_Q1, action_values_Q2) - np.sum(target_pi * np.log(target_pi + 1e-6))
        # END NEW

        value_target_ts = tf.convert_to_tensor(value_target.reshape(-1, 1))
        loss_value = self.model_value.train_on_batch(s_batch, value_target_ts)

        self.Is_nan = np.isnan(loss) + np.isnan(loss_2) + np.isnan(loss_policy) + np.isnan(loss_value)
        return np.all(np.isfinite(loss)) & np.all(np.isfinite(loss_2)) & np.all(np.isfinite(loss_policy)) & \
               np.all(np.isfinite(loss_value))

    def summary(self):
        stringlist = []
        stringlist.append('model_value')
        self.model_value.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_Q')
        self.model_Q.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_policy')
        self.model_policy.summary(print_fn=lambda x: stringlist.append(x))

        short_model_summary = "\n".join(stringlist)
        return short_model_summary
