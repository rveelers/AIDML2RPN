""" Old code!! """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numbers import Number

from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.utils import BaseDeepQ
# from l2rpn_baselines.utils import TrainingParam

from sac_training_param import TrainingParamSAC


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

        # For automatic alpha/temperature tuning.
        self._automatic_alpha_tuning = training_param.AUTOMATIC_ALPHA_TUNING
        if self._automatic_alpha_tuning:
            self._log_alpha = tf.Variable(0.0)
            self._alpha = tfp.util.DeferredTensor(pretransformed_input=self._log_alpha, transform_fn=tf.exp)
            self._alpha_lr = training_param.ALPHA_LR
            self._alpha_optimizer = tf.optimizers.Adam(self._alpha_lr, name='alpha_optimizer')
            # Set the target entropy according to the paper: "SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS"
            # https://arxiv.org/pdf/1910.07207.pdf
            self._target_entropy = 0.98 * (-np.log(1.0 / action_size))

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

        # (1) training of the Q-FUNCTION networks ######################################################################
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

        # (2) training of the POLICY network ###########################################################################
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
        if self._automatic_alpha_tuning:
            temp = self._alpha
        else:
            temp = 1 / np.log(self.life_spent) / 2

        # Calculate a probability distribution over the actions (one distribution for every sample in the batch).
        # Here the temperature parameter comes into play!
        new_proba = np.exp(advantage / temp) / np.sum(np.exp(advantage / temp), axis=-1).reshape(batch_size, 1)
        new_proba_ts = tf.convert_to_tensor(new_proba)

        # The loss function used is the categorical cross-ENTROPY loss = - sum_a (new_proba(a) * log(policy(s, a))
        loss_policy = self.model_policy.train_on_batch(s_batch, new_proba_ts)

        # (3) training of the VALUE FUNCTION network ###################################################################
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

        # (4) tune alpha/temperature parameter #########################################################################
        # This is adapted from the softlearning GitHub-repo:
        # https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py
        if self._automatic_alpha_tuning:
            if not isinstance(self._target_entropy, Number):
                self._target_entropy = 0.0

            log_pis = np.log(new_proba[np.arange(batch_size), a_batch] + 1e-6)

            with tf.GradientTape() as tape:
                alpha_losses = -1.0 * (self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
                # NOTE(hartikainen): It's important that we take the average here, otherwise we end up effectively
                # having `batch_size` times too large learning rate.
                alpha_loss = tf.nn.compute_average_loss(alpha_losses)

            alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
            self._alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._log_alpha]))
            if tf_writer is not None:
                tf.summary.scalar("alpha/log_alpha", self._log_alpha, self.life_spent)
                tf.summary.scalar("alpha/alpha", self._alpha, self.life_spent)

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


class SACAgent(DeepQAgent):

    def __init__(self, action_space):
        super().__init__(action_space)
        self.name = 'SACAgent'
        # Number of environments ...
        self.__nb_env = 1  # TODO: understand why this must be added, as it seems to be part of super().__init__?

    def init_deep_q(self, transformed_observation):
        self.deep_q = SACNetwork(self.action_space.size(),
                                 observation_size=transformed_observation.shape[-1],
                                 lr=self.lr,
                                 learning_rate_decay_rate=self.learning_rate_decay_rate,
                                 learning_rate_decay_steps=self.learning_rate_decay_steps)

    def train(self, env, iterations, save_path, logdir, training_param=TrainingParamSAC()):  # CHANGED
        """ Need to pass TraininParamSAC here, otherwise the class from utils.TrainingParam is used. """

        self.training_param = training_param
        self._init_replay_buffer()

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        self.set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        if logdir is not None:
            self.tf_writer = tf.summary.create_file_writer(logdir, name=self.name)
        else:
            logpath = None
            self.tf_writer = None
        UPDATE_FREQ = 100  # update tensorboard every "UPDATE_FREQ" steps
        SAVING_NUM = 1000

        training_step = 0

        # some parameters have been move to a class named "training_param" for convenience
        self.epsilon = training_param.INITIAL_EPSILON

        # now the number of alive frames and total reward depends on the "underlying environment". It is vector instead
        # of scalar
        alive_frame, total_reward = self._init_global_train_loop()
        reward, done = self._init_local_train_loop()
        epoch_num = 0
        self.losses = np.zeros(iterations)
        alive_frames = np.zeros(iterations)
        total_rewards = np.zeros(iterations)
        new_state = None
        self.reset_num = 0
        with tqdm(total=iterations) as pbar:
            while training_step < iterations:
                # reset or build the environment
                initial_state = self._need_reset(env, training_step, epoch_num, done, new_state)

                # Slowly decay the exploration parameter epsilon
                # if self.epsilon > training_param.FINAL_EPSILON:
                self.epsilon = training_param.get_next_epsilon(current_step=training_step)

                if training_step == 0:
                    # we initialize the NN with the proper shape
                    self.init_deep_q(initial_state)

                # then we need to predict the next moves. Agents have been adapted to predict a batch of data
                pm_i, pq_v, act = self._next_move(initial_state, self.epsilon)

                # todo store the illegal / ambiguous / ... actions
                reward, done = self._init_local_train_loop()
                if self.__nb_env == 1:
                    # still the "hack" to have same interface between multi env and env...
                    # yeah it's a pain
                    act = act[0]

                temp_observation_obj, temp_reward, temp_done, info = env.step(act)
                if self.__nb_env == 1:
                    # dirty hack to wrap them into list
                    temp_observation_obj = [temp_observation_obj]
                    temp_reward = np.array([temp_reward], dtype=np.float32)
                    temp_done = np.array([temp_done], dtype=np.bool)
                    info = [info]
                new_state = self.convert_obs_train(temp_observation_obj)

                self._updage_illegal_ambiguous(training_step, info)
                done, reward, total_reward, alive_frame, epoch_num \
                    = self._update_loop(done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num)

                # update the replay buffer
                # NEW (Johan 2020-06-05): If the following line is not added, s == s2 always in the replay pool
                new_state = new_state.copy()
                # Adding the line above seems to be enough, but better safe than sorry. Let's also add:
                initial_state = initial_state.copy()
                # pm_i.numpy().copy()
                reward = reward.copy()
                done = done.copy()
                # for extra safety.
                # END NEW
                self._store_new_state(initial_state, pm_i, reward, done, new_state)

                # now train the model
                if training_step % self.training_param.NUM_FRAMES == 0:
                    if not self._train_model(training_param, training_step):
                        # infinite loss in this case
                        print("ERROR INFINITE LOSS")
                        break

                # Save the network every 1000 iterations
                if training_step % SAVING_NUM == 0 or training_step == iterations - 1:
                    self.save(save_path)

                # save some information to tensorboard
                alive_frames[epoch_num] = np.mean(alive_frame)
                total_rewards[epoch_num] = np.mean(total_reward)
                self._store_action_played_train(training_step, pm_i)

                self._save_tensorboard(training_step, epoch_num, UPDATE_FREQ, total_rewards, alive_frames)
                training_step += 1
                pbar.update(1)

    def summary(self):
        return self.deep_q.summary()