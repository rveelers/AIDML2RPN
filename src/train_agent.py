import grid2op
import numpy as np
from grid2op.Reward import RedispReward

from hyper_parameters import NUM_FRAMES, INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY, MIN_OBSERVATION, MINIBATCH_SIZE


class TrainAgent(object):
    def __init__(self, agent, reward_fun=RedispReward, env=None):
        self.agent = agent
        self.reward_fun = reward_fun
        self.env = env

    def _build_valid_env(self):
        # now we are creating a valid Environment
        # it's mandatory because no environment are created when the agent is
        # an Agent should not get direct access to the environment, but can interact with it only by:
        # * receiving reward
        # * receiving observation
        # * sending action

        close_env = False

        if self.env is None:
            self.env = grid2op.make(action_class=type(self.agent.action_space({})),
                                    reward_class=self.reward_fun)
            close_env = True

        # I make sure the action space of the user and the environment are the same.
        if not isinstance(self.agent.init_action_space, type(self.env.action_space)):
            raise RuntimeError("Imposssible to build an agent with 2 different action space")
        if not isinstance(self.env.action_space, type(self.agent.init_action_space)):
            raise RuntimeError("Imposssible to build an agent with 2 different action space")

        # A buffer that keeps the last `NUM_FRAMES` images
        self.agent.replay_buffer.clear()
        self.agent.process_buffer = []

        # make sure the environment is reset
        obs = self.env.reset()
        self.agent.process_buffer.append(self.agent.convert_obs(obs))
        do_nothing = self.env.action_space()
        for _ in range(NUM_FRAMES - 1):
            # Initialize buffer with the first frames
            s1, r1, _, _ = self.env.step(do_nothing)
            self.agent.process_buffer.append(self.agent.convert_obs(s1))
        return close_env

    def train(self, num_frames, env=None):
        # this function existed in the original implementation, but has been slightly adapted.

        # first we create an environment or make sure the given environment is valid
        close_env = self._build_valid_env()

        # bellow that, only slight modification has been made. They are highlighted
        observation_num = 0
        curr_state = self.agent.convert_process_buffer()
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" % observation_num))

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY

            initial_state = self.agent.convert_process_buffer()
            self.agent.process_buffer = []

            # it's a bit less convenient that using the SpaceInvader environment.
            # first we need to initiliaze the neural network
            self.agent.init_deep_q(curr_state)
            # then we need to predict the next move
            predict_movement_int, predict_q_value = self.agent.deep_q.predict_movement(curr_state, epsilon)
            # and then we convert it to a valid action
            act = self.agent.convert_act(predict_movement_int)

            reward, done = 0, False
            for i in range(NUM_FRAMES):
                temp_observation_obj, temp_reward, temp_done, _ = self.env.step(act)
                # here it has been adapted too. The observation get from the environment is
                # first converted to vector

                # below this line no changed have been made to the original implementation.
                reward += temp_reward
                self.agent.process_buffer.append(self.agent.convert_obs(temp_observation_obj))
                done = done | temp_done

            if done:
                print("Lived with maximum time ", alive_frame)
                print("Earned a total of reward equal to ", total_reward)
                # reset the environment
                self.env.reset()

                alive_frame = 0
                total_reward = 0

            new_state = self.agent.convert_process_buffer()
            self.agent.replay_buffer.add(initial_state, predict_movement_int, reward, done, new_state)
            total_reward += reward
            if self.agent.replay_buffer.size() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.agent.replay_buffer.sample(MINIBATCH_SIZE)
                self.agent.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.agent.deep_q.target_train()

            # Save the network every 1000 iterations after 50000 iterations
            if observation_num > 50000 and observation_num % 1000 == 0 and self.agent.deep_q.Is_nan == 0:
                print("Saving Network")
                self.agent.deep_q.save_network("saved_agent_" + self.agent.mode + ".h5")

            alive_frame += 1
            observation_num += 1

        if close_env:
            print("closing env")
            self.env.close()

    def calculate_mean(self, num_episode=100, env=None):
        # this method has been only slightly adapted from the original implementation

        # Note that it is NOT the recommended method to evaluate an Agent. Please use "Grid2Op.Runner" instead

        # first we create an environment or make sure the given environment is valid
        close_env = self._build_valid_env()

        reward_list = []
        print("Printing scores of each trial")
        for i in range(num_episode):
            done = False
            tot_award = 0
            self.env.reset()
            while not done:
                state = self.agent.convert_process_buffer()

                # same adapation as in "train" function.
                predict_movement_int = self.agent.deep_q.predict_movement(state, 0.0)[0]
                predict_movement = self.agent.convert_act(predict_movement_int)

                # same adapation as in the "train" funciton
                observation_obj, reward, done, _ = self.env.step(predict_movement)
                observation_vect_full = observation_obj.to_vect()

                tot_award += reward
                self.agent.process_buffer.append(observation_vect_full)
                self.agent.process_buffer = self.agent.process_buffer[1:]
            print(tot_award)
            reward_list.append(tot_award)

        if close_env:
            self.env.close()
        return np.mean(reward_list), np.std(reward_list)