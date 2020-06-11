import numpy as np

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from old_files.deep_q_network import DeepQ
from hyper_parameters import BUFFER_SIZE, FINAL_EPSILON, INITIAL_EPSILON, BATCH_SIZE, NUM_FRAMES
from progress_bar import print_progress
from old_files.replay_buffer import ReplayBuffer


class OldDeepQAgent(AgentWithConverter):

    def __init__(self, action_space):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        self.deep_q = None
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.process_buffer = []
        self.action_history = []
        self.reward_history = []
        self.id = self.__class__.__name__

    def init_deep_q(self, transformed_observation):
        self.deep_q = DeepQ(self.action_space.n, transformed_observation.shape[0])

    def convert_obs(self, observation):
        """ The DeepQ network uses the rho values and line status values as input. """
        # converted_obs = np.concatenate((
        #     observation.prod_p / 50,
        #     observation.load_p / 10,
        #     observation.rho / 2,
        #     observation.timestep_overflow / 10,
        #     observation.line_status,
        #     (observation.topo_vect + 1) / 3,
        #     observation.time_before_cooldown_line / 10,
        #     observation.time_before_cooldown_sub / 10))
        # if np.any(converted_obs > 1) or np.any(converted_obs < 0):
        #     print('out of 0-1 scale')
        converted_obs = np.concatenate((
            observation.prod_p,
            observation.load_p,
            observation.rho,
            observation.timestep_overflow,
            observation.line_status,
            observation.topo_vect,
            observation.time_before_cooldown_line,
            observation.time_before_cooldown_sub))
        return converted_obs

    def my_act(self, transformed_observation, reward, done=False):
        """ This method is called by the environment when using Runner. """
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)

        while len(self.process_buffer) <= NUM_FRAMES:
            self.process_buffer.append(transformed_observation)
        self.process_buffer.pop(0)

        predict_movement_int, _ = self.deep_q.predict_movement(np.concatenate(self.process_buffer), epsilon=0.0)
        self.action_history.append(predict_movement_int)
        # print(self.convert_act(predict_movement_int))
        return predict_movement_int

    def reset_action_history(self):
        self.action_history = []

    def reset_reward_history(self):
        self.reward_history = []

    def save(self, path):
        self.deep_q.save_network(path)

    def load(self, path):
        self.deep_q.load_network(path)

    def train(self, env, num_iterations=10000, network_path=None):
        """ Train the agent. """
        process_buffer = []
        transformed_observation = self.convert_obs(env.reset())

        for _ in range(NUM_FRAMES):
            process_buffer.append(transformed_observation)

        if self.deep_q is None:
            self.init_deep_q(transformed_observation)

        curr_state = np.concatenate(process_buffer)
        epsilon = INITIAL_EPSILON
        epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / (num_iterations * 0.75)
        total_reward = 0
        reset_count = 0
        loss = -1

        for iteration in range(num_iterations):
            print_progress(iteration+1, num_iterations, prefix='Step {}/{}'.format(iteration+1, num_iterations),
                           suffix='Episode count: {}'.format(reset_count))

            # Epsilon becomes smaller over time
            # if epsilon > FINAL_EPSILON:
            #     epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY
            epsilon -= epsilon_decay
            process_buffer = []

            # Predict the next step ...
            predict_movement_int, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)
            act = self.convert_act(predict_movement_int)

            # ... and observe the next step
            reward, done, temp_observation_obj = 0, False, None
            for i in range(NUM_FRAMES):
                if not done:
                    temp_observation_obj, temp_reward, temp_done, _ = env.step(act)
                    reward += temp_reward
                    done = done | temp_done
                process_buffer.append(self.convert_obs(temp_observation_obj))

            # observation, reward, done, _ = env.step(act)
            new_state = np.concatenate(process_buffer)
            self.replay_buffer.add(curr_state, predict_movement_int, reward, done, new_state)
            self.action_history.append(predict_movement_int)
            self.reward_history.append(reward)
            curr_state = new_state
            total_reward += reward

            # reset the environment
            if done:
                transformed_observation = self.convert_obs(env.reset())
                process_buffer = []
                for _ in range(NUM_FRAMES):
                    process_buffer.append(transformed_observation)
                curr_state = np.concatenate(process_buffer)
                reset_count += 1

            # Start training the network when the replay buffer is full and train each specific number of steps
            if iteration > BATCH_SIZE:  # and iteration % TRAIN_INTERVAL == 0:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(BATCH_SIZE)
                loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch)
                self.deep_q.target_train()

            # Replace the target network each specific number of steps
            # if iteration % REPLACE_TARGET_INTERVAL == 0:
            #     self.deep_q.replace_target()

            # Save the network every 1000 iterations and final iteration
            if iteration % 1000 == 999 or iteration == num_iterations-1:
                print("Saving Network, current loss:", loss)
                self.deep_q.save_network(network_path)

        env.close()
