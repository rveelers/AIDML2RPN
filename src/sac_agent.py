import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

# from sac_network import SACNetwork
from sac_new import SACNetwork
from sac_training_param import TrainingParamSAC
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from l2rpn_baselines_old.utils.ReplayBuffer import ReplayBuffer


class SACAgent(AgentWithConverter):
    def __init__(self, action_space, name='SACAgent', training_param=TrainingParamSAC(), store_action=False,
                 istraining=False):

        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)
        self.training_param = training_param
        self.name = name
        self.store_action = store_action
        self.istraining = istraining

        self.epsilon = training_param.INITIAL_EPSILON

        self.replay_buffer = ReplayBuffer(training_param.BUFFER_SIZE)
        self.deep_q = None  # Initialise when we know size of observation
        self.tf_writer = None
        self.graph_saved = False

        # Statistics
        self.losses = None
        self.epoch_num_steps_alive = None
        self.epoch_rewards = None
        self.reset_num = None

        self.dict_action = {}

        self.actions_per_1000steps = np.zeros((1000, self.action_space.size()), dtype=np.int)
        self.illegal_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self.ambiguous_actions_per_1000steps = np.zeros(1000, dtype=np.int)

        self.obs_as_vect = None
        self._tmp_obs = None

    def init_deep_q(self, transformed_observation):
        self.deep_q = SACNetwork(self.action_space.size(),
                                 observation_size=transformed_observation.shape[-1],
                                 training_param=self.training_param)

    def my_act(self, transformed_observation, reward, done=False):
        """ Used when evaluating the agent """
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, probs = self.deep_q.predict_movement(transformed_observation)
        res = int(predict_movement_int)
        return res

    def my_acts(self, transformed_observation, nr_acts):
        """ Used when evaluating the agent """
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        acts, probs = self.deep_q.predict_movement_evaluate(transformed_observation, nr_acts)
        return acts, probs

    def _next_move(self, curr_state, epsilon=0.0):
        """ Used in training. Includes exploration"""
        # action, prob = self.deep_q.predict_movement_stochastic(curr_state)
        action, _ = self.deep_q.predict_movement(curr_state, epsilon=epsilon)
        return int(action)

    def convert_obs(self, observation):
        tmp = np.concatenate((
            observation.prod_p / 150,
            observation.load_p / 120,
            observation.rho / 2,
            observation.timestep_overflow / 10,
            observation.line_status,
            (observation.topo_vect + 1) / 3,
            observation.time_before_cooldown_line / 10,
            observation.time_before_cooldown_sub / 10)).reshape(1, -1)

        if self._tmp_obs is None:
            self._tmp_obs = np.zeros((1, tmp.shape[1]), dtype=np.float32)
        else:
            self._tmp_obs[:] = tmp
        return self._tmp_obs

    def train(self, env, iterations, save_path, logdir, training_param):
        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        self.set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        # Tensorboard writer
        if logdir is not None:
            self.tf_writer = tf.summary.create_file_writer(logdir, name=self.name)

        self.losses = np.zeros(iterations)
        self.epoch_num_steps_alive = np.zeros(iterations)
        self.epoch_rewards = np.zeros(iterations)
        self.reset_num = 0

        # Initialize the NN with proper shape
        obs = env.reset()
        self.init_deep_q(self.convert_obs(obs))

        epoch_num = 0  # Increase this every time we reach the done state
        with tqdm(total=iterations) as pbar:
            for training_step in range(iterations):
                # Get current/initial state
                initial_state = self.convert_obs(obs)

                # Slowly decay the exploration parameter epsilon
                self.epsilon = training_param.get_next_epsilon(current_step=training_step)

                # predict next moves with epsilon chance of random action
                act = self._next_move(initial_state, epsilon=self.epsilon)

                # Take step and convert obs
                act_obj = self.convert_act(act)
                obs, reward, done, info = env.step(act_obj)
                new_state = self.convert_obs(obs)

                if done:
                    epoch_num += 1
                    reward = 0
                    obs = env.reset()
                else:
                    self.epoch_num_steps_alive[epoch_num] += 1
                    self.epoch_rewards[epoch_num] += reward

                # Add to replay buffer
                self.replay_buffer.add(initial_state.squeeze().copy(), act, reward, done, new_state.squeeze().copy())

                self._store_action_played_train(training_step, act)
                self._update_illegal_ambiguous(training_step, info)

                # Train the model
                if not self._train_model(training_param, training_step):
                    print("ERROR INFINITE LOSS")
                    break

                # Save network every SAVING_NUM steps
                if training_step % self.training_param.SAVING_NUM == 0 or training_step == iterations - 1:
                    self.deep_q.save_network(save_path)

                # Save stats to Tensorboard every UPDATE_FREQ steps
                if training_step % self.training_param.UPDATE_FREQ == 0 and epoch_num > 0:
                    self._save_tensorboard(epoch_num, training_step, self.epoch_rewards, self.epoch_num_steps_alive)

                # Update progress bar
                pbar.update(1)

    def _train_model(self, training_param, training_step):
        losses_are_all_finite = True

        if training_step > max(training_param.MIN_OBSERVATION, training_param.MINIBATCH_SIZE):
            # Get batch of training samples from buffer
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(training_param.MINIBATCH_SIZE)

            # Train on batch
            losses_are_all_finite = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, self.tf_writer)

            # Update target Q networks
            self.deep_q.target_train()

        return losses_are_all_finite

    def _save_tensorboard(self, epoch_num, training_step, epoch_rewards, epoch_alive):
        if self.tf_writer is None:
            return

        with self.tf_writer.as_default():
            last_alive = epoch_alive[(epoch_num - 1)]  # Number of steps we stayed alive in the previous epoch
            last_reward = epoch_rewards[(epoch_num - 1)]  # Total reward for previous epoch

            mean_reward = np.nanmean(epoch_rewards[:epoch_num])  # Average epoch reward
            mean_alive = np.nanmean(epoch_alive[:epoch_num])  # Average num steps alive/epoch

            tmp = self.actions_per_1000steps > 0
            tmp = tmp.sum(axis=0)
            nb_action_taken_last_1000_step = np.sum(tmp > 0)  # number of different actions taken last 1000 steps
            nb_illegal_act = np.sum(self.illegal_actions_per_1000steps)  # number of illegal acts/1000 steps
            nb_ambiguous_act = np.sum(self.ambiguous_actions_per_1000steps)  # number of ambiguous acts/1000 steps

            if epoch_num >= 30:
                mean_reward_30 = np.nanmean(epoch_rewards[(epoch_num - 30):epoch_num])
                mean_alive_30 = np.nanmean(epoch_alive[(epoch_num - 30):epoch_num])
            else:
                mean_reward_30 = mean_reward
                mean_alive_30 = mean_alive

            # show first the Mean reward and mine time alive (hence the upper case)
            tf.summary.scalar("length_of_epochs/last_epoch", last_alive, training_step)
            tf.summary.scalar("length_of_epochs/mean_30", mean_alive_30, training_step)
            tf.summary.scalar("length_of_epochs/total_mean", mean_alive, training_step)

            tf.summary.scalar("rewards/total_reward_last_epoch", last_reward, training_step)
            tf.summary.scalar("rewards/mean_reward_30_epochs", mean_reward_30, training_step)
            tf.summary.scalar("rewards/mean_reward_all_epochs", mean_reward, training_step)

            tf.summary.scalar("actions/nb_differentaction_taken_1000", nb_action_taken_last_1000_step, training_step)
            tf.summary.scalar("actions/nb_illegal_act_1000", nb_illegal_act, training_step)
            tf.summary.scalar("actions/nb_ambiguous_act_1000", nb_ambiguous_act, training_step)

    # utilities for data reading
    def set_chunk(self, env, nb):
        env.set_chunk_size(int(max(100, nb)))

    def _update_illegal_ambiguous(self, curr_step, info):
        self.illegal_actions_per_1000steps[curr_step % 1000] = int(info["is_illegal"])
        self.ambiguous_actions_per_1000steps[curr_step % 1000] = int(info["is_ambiguous"])

    def _store_action_played_train(self, training_step, action_id):
        which_row = training_step % 1000
        self.actions_per_1000steps[which_row, :] = 0
        self.actions_per_1000steps[which_row, action_id] += 1


