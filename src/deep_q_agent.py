import os

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from deep_q_network import DeepQ
from hyper_parameters import BUFFER_SIZE, FINAL_EPSILON, INITIAL_EPSILON, BATCH_SIZE, TRAIN_INTERVAL, \
    REPLACE_TARGET_INTERVAL
from progress_bar import print_progress
from replay_buffer import ReplayBuffer


class DeepQAgent(AgentWithConverter):

    def __init__(self, action_space, observation_size, network=DeepQ):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        self.action_size = action_space.size()
        self.observation_size = observation_size
        self.network_name = network.__name__
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.deep_q = network(self.action_size, self.observation_size)
        self.action_history = []

    def convert_obs(self, observation):
        """ The DeepQ network uses the rho values of the lines as input. """
        return observation.rho

    def my_act(self, transformed_observation, reward, done=False):
        """ This method is called by the environment when using Runner. """
        predict_movement_int, _ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        self.action_history.append(predict_movement_int)
        # print(self.convert_act(predict_movement_int))
        return predict_movement_int

    def save_network(self, path):
        self.deep_q.save_network(path)

    def load_network(self, path):
        self.deep_q.load_network(path)

    def train(self, env, num_iterations=10000):
        """ Train the agent. """
        curr_state = self.convert_obs(env.reset())
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

            # Predict the next step ...
            predict_movement_int, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)
            act = self.convert_act(predict_movement_int)

            # ... and observe the next step
            observation, reward, done, _ = env.step(act)
            new_state = self.convert_obs(observation)

            self.replay_buffer.add(curr_state, predict_movement_int, reward, done, new_state)
            curr_state = new_state
            total_reward += reward

            # reset the environment
            if done:
                # print("Lived with maximum time ", alive_frame)
                # print("Earned a total of reward equal to ", total_reward)
                env.reset()
                reset_count += 1

            # Start training the network when the replay buffer is full and train each specific number of steps
            if iteration > BUFFER_SIZE and iteration % TRAIN_INTERVAL == 0:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(BATCH_SIZE)
                loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch)

            # Replace the target network each specific number of steps
            if iteration % REPLACE_TARGET_INTERVAL == 0:
                self.deep_q.replace_target()

            # Save the network every 1000 iterations and final iteration
            if iteration % 1000 == 999 or iteration == num_iterations-1:
                print("\nSaving Network, current loss:", loss)
                network_path = os.path.join('saved_networks', '{}_{}_{}.h5'.format(
                    env.name, self.network_name, num_iterations))
                self.deep_q.save_network(network_path)

        env.close()
