""" In this file, SACBaselineAgent and SACAgent is defined. The latter is our implementation. """

import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

from l2rpn_baselines.SAC import SAC  # <-- baseline SAC agent, very similar to DeepQAgent

from sac_network import SACNetwork
from sac_training_param import TrainingParamSAC


class SACBaselineAgent(SAC):
    """ This is essentially the baseline SAC agent, but with the buffer error fixed. The SAC baseline agent is
     essentially DeepQAgent, initialized with a different network. The train function in DeepQAgent is where the
     problem is. """

    def __init__(self, action_space):
        super().__init__(action_space)
        self.name = 'SACBaselineAgent'
        self.__nb_env = 1  # TODO: understand why this must be added, as it seems to be part of super().__init__?

    def train(self, env, iterations, save_path, logdir, training_param=TrainingParamSAC()):  # NEW Change (1)
        """ Three changes: (1) Use TrainingParamSAC instead of TrainingParam (line above), (2) make a small change to
        where the logs are saved, (3) fix buffer error. """

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
            self.tf_writer = tf.summary.create_file_writer(logdir, name=self.name)  # NEW Change (2)
        else:
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
                # Change (3)
                new_state = new_state.copy()
                # Adding the line above seems to be enough, but better safe than sorry. Let's also add:
                initial_state = initial_state.copy()
                pm_i.copy()
                reward = reward.copy()
                done = done.copy()
                # for extra safety.
                # END NEW
                self._store_new_state(initial_state, pm_i, reward, done, new_state)

                # now train the model
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


class SACAgent(SACBaselineAgent):
    def __init__(self, action_space):
        super().__init__(action_space)
        self.name = 'SACAgent'

    #def summary(self):
    #    return self.deep_q.summary()

    def init_deep_q(self, transformed_observation):
        self.deep_q = SACNetwork(self.action_space.size(),
                                 observation_size=transformed_observation.shape[-1],
                                 lr=self.lr,
                                 learning_rate_decay_rate=self.learning_rate_decay_rate,
                                 learning_rate_decay_steps=self.learning_rate_decay_steps)

    def _train_model(self, training_param, training_step):
        losses_are_all_finite = True

        if training_step > max(training_param.MIN_OBSERVATION, training_param.MINIBATCH_SIZE):
            # Get batch of training samples from buffer
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(training_param.MINIBATCH_SIZE)

            # Train on batch
            losses_are_all_finite = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, self.tf_writer)

            # save learning rate for later
            self.train_lr = self.deep_q.optimizer_Q._decayed_lr('float32').numpy()

            # Update target Q networks
            self.deep_q.target_train()

        return losses_are_all_finite


