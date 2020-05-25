import os

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from deep_q_network import DeepQ
from hyper_parameters import BUFFER_SIZE, FINAL_EPSILON, INITIAL_EPSILON, EPSILON_DECAY, BATCH_SIZE, TRAIN_INTERVAL
from progress_bar import print_progress
from replay_buffer import ReplayBuffer


class DeepQAgent(AgentWithConverter):

    def __init__(self, action_space, observation_size, network=DeepQ):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        self.action_size = action_space.size()
        self.observation_size = observation_size
        self.network = network
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.deep_q = network(self.action_size, self.observation_size)

    def convert_obs(self, observation):
        return observation.rho

    def my_act(self, transformed_observation, reward, done=False):
        predict_movement_int, _ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        return predict_movement_int

    def save_network(self, path):
        self.deep_q.save_network(path)

    def load_network(self, path):
        self.deep_q.load_network(path)

    def train(self, env, num_iterations=10000):
        curr_state = self.convert_obs(env.reset())
        epsilon = INITIAL_EPSILON
        total_reward = 0
        reset_count = 0

        for iteration in range(num_iterations):
            print_progress(iteration+1, num_iterations, prefix='Step {}/{}'.format(iteration+1, num_iterations),
                           suffix='Episode count: {}'.format(reset_count))

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY

            predict_movement_int, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)
            act = self.convert_act(predict_movement_int)

            observation, reward, done, _ = env.step(act)
            new_state = self.convert_obs(observation)

            # reset the environment
            if done:
                # print("Lived with maximum time ", alive_frame)
                # print("Earned a total of reward equal to ", total_reward)
                env.reset()
                reset_count += 1

            self.replay_buffer.add(curr_state, predict_movement_int, reward, done, new_state)
            curr_state = new_state
            total_reward += reward
            loss = -1

            # Start training the network when the replay buffer is full and train each specific number of steps
            if iteration > BUFFER_SIZE and iteration % TRAIN_INTERVAL == 0:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(BATCH_SIZE)
                loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch)
                self.deep_q.target_train()

            # Save the network every 1000 iterations and final iteration
            if iteration > BUFFER_SIZE and iteration % 1000 == 999 or iteration == num_iterations-1:
                print("Saving Network, current loss:", loss)
                network_path = os.path.join('saved_networks', 'agent_{}_{}_{}.h5'.format(env.name, self.network, num_iterations))
                self.deep_q.save_network(network_path)

        env.close()
