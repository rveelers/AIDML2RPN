import os
import time

import tensorflow as tf
import numpy as np

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from deep_q_network import DeepQ
from hyper_parameters import BUFFER_SIZE, FINAL_EPSILON, INITIAL_EPSILON, BATCH_SIZE
from progress_bar import print_progress
from replay_buffer import ReplayBuffer


class DeepQAgent(AgentWithConverter):
    """ Agent using a Deep Q-network as an approximator of the Q-function. """

    def __init__(self, action_space):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        self.deep_q = None
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.process_buffer = []
        self.id = self.__class__.__name__

        # Statistics
        self.action_history = []
        self.reward_history = []
        self.cumulative_reward = 0
        self.smallest_loss = np.inf
        self.run_step_count = 0
        self.run_tf_writer = None

    def init_deep_q(self, transformed_observation):
        self.deep_q = DeepQ(self.action_space.n, transformed_observation.shape[0])

    def convert_obs(self, observation):
        """
        This method converts an CompleteObservation object from Grid2Op to a vector that can be inputted in the
        DeepQ network.

        The vector contains the generator production values, consumer load values, the ratio between
        line loads and its thermal limits (rho value) and line status values as input. All values are rescaled by
        dividing by a constant factor. It is hardcoded and the values are found by observing maximum values seen in the
         environment. A more dynamic way of normalizing would be preferred.
         """
        converted_obs = np.concatenate((
            observation.prod_p / 150,
            observation.load_p / 120,
            observation.rho / 2,
            observation.timestep_overflow / 10,
            observation.line_status,
            (observation.topo_vect + 1) / 3,
            observation.time_before_cooldown_line / 10,
            observation.time_before_cooldown_sub / 10))
        return converted_obs

    def my_act(self, transformed_observation, reward, done=False, obs=None, allow_actions_once=False):
        """ This method is called to decide what action should be taken in that step.
        It is called by the Grid2Op runner as well. """

        if self.deep_q is None:
            self.init_deep_q(transformed_observation)

        # ---- Get action -----

        if obs is None:
            best_action, _ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        else:
            predicted_qvalues = self.deep_q.predict_rewards(transformed_observation)
            sorted_qvalues = np.argsort(predicted_qvalues)[::-1]
            if allow_actions_once:
                sorted_qvalues = [value for value in sorted_qvalues if value not in self.action_history]
            best_actions = sorted_qvalues[:4]
            print(best_actions)
            best_action, best_reward = 0, 0.
            _, best_reward, _, _ = obs.simulate(self.convert_act(best_action))
            for action in best_actions:
                _, expected_reward, _, _ = obs.simulate(self.convert_act(action))
                if expected_reward > best_reward:
                    best_action = action
                    best_reward = expected_reward

        # ---- Tensorboard statistics ----

        if self.run_tf_writer is None:
            log_path = os.path.join('logs', 'run', self.id + '_' + str(time.time()))
            self.run_tf_writer = tf.summary.create_file_writer(log_path)

        self.action_history.append(best_action)
        self.cumulative_reward += reward

        with self.run_tf_writer.as_default():
            tf.summary.scalar("action", best_action, self.run_step_count)
            tf.summary.scalar("reward", reward, self.run_step_count)
            tf.summary.scalar("cumulative reward", self.cumulative_reward, self.run_step_count)

        self.run_step_count += 1

        return best_action

    def reset_action_history(self):
        self.action_history = []

    def reset_reward_history(self):
        self.reward_history = []

    def save(self, path):
        self.deep_q.save_network(path)

    def load(self, path):
        self.deep_q.load_network(path)

    def train(self, env, num_iterations=10000, network_path=None):
        """ Train the agent.

        The agent runs in the environment for a specified number of iterations. When the done state is reached the
        environment is reset. When their are enough samples in the replay buffer. Training of the network starts.
        """

        # Initialize Tensorboard writer
        log_path = os.path.join('logs', 'train', self.id + '_' + str(time.time()))
        tf_writer = tf.summary.create_file_writer(log_path)

        transformed_observation = self.convert_obs(env.reset())
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)

        epsilon = INITIAL_EPSILON
        epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / num_iterations
        total_reward = 0
        reset_count = 0
        current_loss = np.inf

        for iteration in range(num_iterations):
            print_progress(iteration+1, num_iterations, prefix='Step {}/{}'.format(iteration+1, num_iterations),
                           suffix='Episode count: {}'.format(reset_count))

            # Decay epsilon over time
            epsilon -= epsilon_decay

            # Predict the next step ...
            curr_state = self.convert_obs(env.get_obs())
            predict_movement_int, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)
            act = self.convert_act(predict_movement_int)

            # ... and observe the action
            observation, reward, done, _ = env.step(act)
            new_state = self.convert_obs(env.get_obs())
            self.replay_buffer.add(curr_state, predict_movement_int, reward, done, new_state)
            self.action_history.append(predict_movement_int)
            self.reward_history.append(reward)
            total_reward += reward

            # reset the environment
            if done:
                env.reset()
                reset_count += 1

            # Start training the network when the replay buffer is large enough to sample batches from
            if iteration > BATCH_SIZE:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(BATCH_SIZE)
                current_loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch)
                self.deep_q.target_train()

            # Save the network every 100 iterations
            if iteration % 100 == 99:
                self.smallest_loss = current_loss
                print("Saving Network, current loss:", current_loss)
                self.deep_q.save_network(network_path)

            # Write to tensorboard
            with tf_writer.as_default():
                tf.summary.scalar("loss", current_loss, iteration)
                tf.summary.scalar("action", predict_movement_int, iteration)
                tf.summary.scalar("reward", reward, iteration)
                tf.summary.scalar("max q-value", predict_q_value, iteration)

        env.close()
