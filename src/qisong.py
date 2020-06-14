import os
import sys
import numpy as np
import random
import warnings
import threading
import time

import grid2op
from grid2op import make
from grid2op.Agent import RandomAgent

from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
import pdb

from collections import deque

from grid2op.Reward import L2RPNReward
from grid2op.Reward import RedispReward
from grid2op.Reward import EconomicReward

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model, Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, subtract, add
    from tensorflow.keras.layers import Input, Lambda, Concatenate

class Action_classfication(AgentWithConverter):
    def __init__(self, env):
        AgentWithConverter.__init__(self, env.action_space, action_space_converter=IdToAct)
        self.env = env
        self.actions_level_1 = []
        self.actions_level_2 = []
        self.all_actions = []

    def my_act(self, transformed_observation, reward, done=False):
        return 0

    def get_actions_levels(self):
      for action in range(env.action_space.size()):  # range(351):
        self.all_actions.append(action)
        real_action = self.convert_act(action)
        if sum(real_action._redispatch) > 0 or sum(real_action._set_line_status) > 0 or sum(real_action._switch_line_status) > 0:
          self.actions_level_2.append(action)
        else:
          self.actions_level_1.append(action)

class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        """Add an experience to the buffer"""
        # S represents current state, a is action,
        # r is reward, d is whether it is the end,
        # and s2 is next state
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class SAC(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size, observation_size, lr=1e-5):
        self.action_size = action_size
        self.observation_size = observation_size
        self.model = None
        self.target_model = None
        self.lr_ = lr
        self.average_reward = 0
        self.life_spent = 1
        #self.qvalue_evolution = []
        self.Is_nan = False

        self.construct_q_network()

    def build_q_NN(self):
        model = Sequential()

        input_states = Input(shape = (self.observation_size,))
        input_action = Input(shape = (self.action_size,))
        input_layer = Concatenate()([input_states, input_action])

        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(500)(lay2)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(1, activation = 'linear')(lay3)

        model = Model(inputs=[input_states, input_action], outputs=[advantage])
        model.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        return model

    def construct_q_network(self):
        #construct double Q networks
        self.model_Q = self.build_q_NN()
        self.model_Q2 = self.build_q_NN()


        #state value function approximation
        self.model_value = Sequential()

        input_states = Input(shape = (self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay3 = Dense(500)(lay1)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(self.action_size, activation = 'relu')(lay3)
        state_value = Dense(1, activation = 'linear')(advantage)

        self.model_value = Model(inputs=[input_states], outputs=[state_value])
        self.model_value.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        self.model_value_target = Sequential()

        input_states = Input(shape = (self.observation_size,))

        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay3 = Dense(500)(lay1)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(self.action_size, activation = 'relu')(lay3)
        state_value = Dense(1, activation = 'linear')(advantage)

        self.model_value_target = Model(inputs=[input_states], outputs=[state_value])
        self.model_value_target.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.model_value_target.set_weights(self.model_value.get_weights())
        #policy function approximation

        self.model_policy = Sequential()
        input_states = Input(shape = (self.observation_size,))
        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(500)(lay2)
        lay3 = Activation('relu')(lay3)

        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)

        self.model_policy = Model(inputs=[input_states], outputs=[soft_proba])
        self.model_policy.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr_))

        print("Successfully constructed networks.")



    def predict_movement_evaluate(self, states):
        """Predict movement of game controler where is epsilon
        probability randomly move."""

        """The process to choose an action is changed."""

        p_actions = self.model_policy.predict(states.reshape(1, self.observation_size)).ravel()

        opt_policy = p_actions.argsort()[-3:][::-1]

        return opt_policy


    def predict_movement(self, states, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""

        """The process to choose an action is changed."""

        #p_actions = self.model_policy.predict(states.reshape(1, self.observation_size)).ravel()

        #opt_policy = np.random.choice(a=self.action_size, size=1, replace = True, p=p_actions)


        # nothing has changed from the original implementation
        p_actions = self.model_policy.predict(states.reshape(1, self.observation_size)).ravel()
        rand_val = np.random.random()

        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.action_size)
        else:
            #opt_policy = np.random.choice(np.arange(NUM_ACTIONS, dtype=int), size=1, p = p_actions)
            opt_policy = np.argmax(p_actions)


        return np.int(opt_policy)

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains networks to fit given parameters"""

        # nothing has changed from the original implementation, except for changing the input dimension 'reshape'
        batch_size = s_batch.shape[0]
        target = np.zeros((batch_size, 1))
        new_proba = np.zeros((batch_size, self.action_size))
        last_action=np.zeros((batch_size, self.action_size))

        #training of the action state value networks
        last_action=np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            last_action[i,a_batch[i]] = 1

            v_t = self.model_value_target.predict(s_batch[i].reshape(1, self.observation_size*NUM_FRAMES), batch_size = 1)

            #self.qvalue_evolution.append(v_t[0])
            fut_action = self.model_value_target.predict(s2_batch[i].reshape(1, self.observation_size*NUM_FRAMES), batch_size = 1)

            target[i,0] = r_batch[i] + (1 - d_batch[i]) * DECAY_RATE * fut_action


        loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        loss_2 = self.model_Q2.train_on_batch([s_batch, last_action], target)

        #training of the policy

        for i in range(batch_size):
            self.life_spent += 1
            temp =  1 / np.log(self.life_spent) / 2
            new_values = self.model_Q.predict([np.tile(s_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                                      np.eye(self.action_size)]).reshape(1,-1)
            new_values -= np.amax(new_values, axis=-1)
            new_proba[i] = np.exp(new_values / temp) / np.sum(np.exp(new_values / temp), axis=-1)

        loss_policy = self.model_policy.train_on_batch(s_batch, new_proba)

        #training of the value_function
        value_target = np.zeros(batch_size)
        for i in range(batch_size):
            target_pi = self.model_policy.predict(s_batch[i].reshape(1, self.observation_size*NUM_FRAMES), batch_size = 1)
            action_v1 = self.model_Q.predict([np.tile(s_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                                      np.eye(self.action_size)]).reshape(1,-1)
            action_v2 = self.model_Q2.predict([np.tile(s_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                                      np.eye(self.action_size)]).reshape(1,-1)
            value_target[i] = np.fmin(action_v1[0,a_batch[i]], action_v2[0,a_batch[i]]) - np.sum(target_pi * np.log(target_pi + 1e-6))

        loss_value = self.model_value.train_on_batch(s_batch, value_target.reshape(-1,1))

        self.Is_nan = np.isnan(loss) + np.isnan(loss_2) + np.isnan(loss_policy) + np.isnan(loss_value)
        # Print the loss every 100 iterations.
        #if observation_num % 100 == 0:
        #    print("We had a loss equal to ", loss, loss_2, loss_policy, loss_value)
        print('Q-losses:', loss, loss_2, '\tPolicy loss:', loss_policy, '\tvalue_loss:', loss_value)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model_policy.save("policy_"+path)
        self.model_value_target.save("value_"+path)
        print("Successfully saved network.")

    def load_network(self, path):
        # nothing has changed
        self.model_policy = load_model("policy_"+path)
        self.model_value_target = load_model("value_"+path)
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model_value.get_weights()
        target_model_weights = self.model_value_target.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.model_value_target.set_weights(model_weights)

class DeepQAgent(AgentWithConverter):
    # first change: An Agent must derived from grid2op.Agent (in this case MLAgent, because we manipulate vector instead
    # of classes)

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        return predict_movement_int

    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            # the first time an observation is observed, I set up the neural network with the proper dimensions.
            if self.mode == "SAC":
                cls = SAC
            else:
                raise RuntimeError("Unknown neural network named \"{}\"".format(self.mode))

            """Justify whether we use a smaller action space or real action space """
            if self.replace_action_space:
                self.deep_q = cls(len(self.new_int_action_space), observation_size=transformed_observation.shape[0], lr=self.lr)
            else:
                self.deep_q = cls(self.action_space.size(), observation_size=transformed_observation.shape[0], lr=self.lr)

    def __init__(self, action_space, mode="SAC", lr=1e-5, replace_action_space = False, new_int_action_space = None):
        # this function has been adapted.
        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        """We make change in the replay buffer"""
        #self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # compare to original implementation, i don't know the observation space size.
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.deep_q = None
        self.mode = mode
        self.lr=lr

        self.replace_action_space = replace_action_space
        self.new_int_action_space = new_int_action_space

    def load_network(self, path):
        # not modified compare to original implementation
        self.deep_q.load_network(path)

class TrainAgent(threading.Thread):
    def __init__(self, level_flag, agent, reward_fun=RedispReward, env=None, another_int_action_space=None):
        threading.Thread.__init__(self)
        self.agent = agent
        self.reward_fun = reward_fun
        self.env = env
        self.level_flag = level_flag
        self.another_int_action_space = another_int_action_space
        self.reward_episode = []
        self.train_not = True
        self.evaluate_reward_list = []
        self.transfer_not = False


    def run(self):
        # this function existed in the original implementation, but has been slightly adapted.

        # first we create an environment or make sure the given environment is valid

        # bellow that, only slight modification has been made. They are highlighted
        observation_num = 0
        self.env.reset()
        state_real = self.env.reset()
        state_vect = state_real.to_vect()
        epsilon = INITIAL_EPSILON
        ''''''
        self.agent.init_deep_q(state_vect)
        ''''''
        alive_frame = 0
        total_reward = 0

        time_start = time.time()

        while self.train_not:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" %observation_num))

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            """Map the output action to the real action space, state is changed """
            act_int = self.agent.deep_q.predict_movement(state_vect, epsilon)
            # and then we convert it to a valid action
            if self.agent.replace_action_space:
                predict_movement_int = self.agent.new_int_action_space[act_int]
            else:
                predict_movement_int = act_int
            act = self.agent.convert_act(predict_movement_int)

            new_state_real, reward, done, _ = self.env.step(act)

            _reward = reward

            if done:
                self.reward_episode.append(total_reward)
                #print("Lived with maximum time ", alive_frame)
                #print("Earned a total of reward equal to ", total_reward)
                # reset the environment
                self.env.reset()
                new_state_real = self.env.reset()
                _reward = -1000
                alive_frame = 0
                total_reward = 0

            new_state_vect = new_state_real.to_vect()

            """Make change in the replay buffer beginning"""
            if self.level_flag == '1_1':
                replay_buffer_1_1.add(state_vect, act_int, _reward, done, new_state_vect)

            if self.level_flag == '2_1' or self.level_flag == '2_2' or self.level_flag == '2_3':
                replay_buffer_2.add(state_vect, act_int, _reward, done, new_state_vect)

            if self.level_flag == '3_1' or self.level_flag == '3_2' or self.level_flag == '3_3':
                replay_buffer_3.add(state_vect, act_int, _reward, done, new_state_vect)

            if self.level_flag == '1_2' or self.level_flag == '1_3':
                if sum(act._redispatch) > 0 or sum(act._set_line_status) > 0 or sum(act._switch_line_status) > 0:
                    replay_buffer_1_2.add(state_vect, act_int, _reward, done, new_state_vect)
                else:
                    replay_buffer_1_2.add(state_vect, act_int, _reward, done, new_state_vect)
                    replay_buffer_1_1.add(state_vect, self.another_int_action_space.index(act_int), _reward, done, new_state_vect)
            """Make change in the replay buffer end """

            total_reward += reward
            state_real = new_state_real
            state_vect = new_state_vect


            """Make change in the batch     beginning"""
            if self.level_flag == '1_1' and replay_buffer_1_1.size() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_buffer_1_1.sample(MINIBATCH_SIZE)
                self.agent.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.agent.deep_q.target_train()
            if (self.level_flag == '1_2' or self.level_flag == '1_3') and replay_buffer_1_2.size() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_buffer_1_2.sample(MINIBATCH_SIZE)
                self.agent.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.agent.deep_q.target_train()
            if (self.level_flag == '2_1' or self.level_flag == '2_2' or self.level_flag == '2_3') and replay_buffer_2.size() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_buffer_2.sample(MINIBATCH_SIZE)
                self.agent.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.agent.deep_q.target_train()
            if (self.level_flag == '3_1' or self.level_flag == '3_2' or self.level_flag == '3_3') and replay_buffer_3.size() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_buffer_3.sample(MINIBATCH_SIZE)
                self.agent.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.agent.deep_q.target_train()
            """Make change in the batch  end"""


            # Save the network every 1000 iterations after 50000 iterations
            if observation_num > 50000 and observation_num % 1000 == 0 and self.agent.deep_q.Is_nan == 0:
                print("Saving Network")
                self.agent.deep_q.save_network("saved_agent_"+self.agent.mode+".h5")

            #print(observation_num, self.level_flag)
            alive_frame += 1
            observation_num += 1
            if time.time() - time_start > Time_Training:
                self.train_not = False
                self.evaluate_reward_list = self.evaluate()

    def evaluate(self, num_episode = 10, env=None):
        # this method has been only slightly adapted from the original implementation

        # Note that it is NOT the recommended method to evaluate an Agent. Please use "Grid2Op.Runner" instead

        # first we create an environment or make sure the given environment is valid
        print("step into evaluation")

        reward_list = []
        #print("Printing scores of each trial")
        for i in range(num_episode):
            done = False
            tot_award = 0
            state_real = self.env.reset()
            while not done:
                state_vect = state_real.to_vect()

                # same adapation as in "train" function.
                """Map the output action to the real action space(int) """
                policy_chosen_list = self.agent.deep_q.predict_movement_evaluate(state_vect)
                policy_chosen_list = np.append(0, policy_chosen_list)
                if self.agent.replace_action_space:
                    _new_int_action_space = np.array(self.agent.new_int_action_space)
                    policy_chosen_list = _new_int_action_space[policy_chosen_list]

                obs_0, rw_0, done_0, _  = state_real.simulate(self.agent.convert_act(policy_chosen_list[0]))
                obs_1, rw_1, done_1, _  = state_real.simulate(self.agent.convert_act(policy_chosen_list[1]))
                obs_2, rw_2, done_2, _  = state_real.simulate(self.agent.convert_act(policy_chosen_list[2]))
                obs_3, rw_3, done_3, _  = state_real.simulate(self.agent.convert_act(policy_chosen_list[3]))

                act_int = policy_chosen_list[np.argmax([rw_0, rw_1, rw_2, rw_3])]
                if act_int != 0:
                    print(act_int)

                predict_movement = self.agent.convert_act(act_int)

                # same adapation as in the "train" funciton
                state_real, reward, done, _ = self.env.step(predict_movement)
                tot_award += reward
            reward_list.append(tot_award)

        return reward_list

class MLRL_SAC_2(object):
    def __init__(self, new_int_action_space):
        self.new_int_action_space_1 = new_int_action_space[0]
        self.new_int_action_space_2 = new_int_action_space[1]

        self.reward_episode_record_1_1 = []
        # self.reward_episode_record_1_2 = []
        # self.reward_episode_record_1_3 = []
        # self.reward_episode_record_2_1 = []
        # self.reward_episode_record_2_2 = []
        # self.reward_episode_record_2_3 = []
        # self.reward_episode_record_3_1 = []
        # self.reward_episode_record_3_2 = []
        # self.reward_episode_record_3_3 = []

        self.transfer_record = []

        self.env_1_1 = make("rte_case14_redisp")
        # self.env_1_2 = make("rte_case14_redisp")
        # self.env_1_3 = make("rte_case14_redisp")
        # self.env_2_1 = make("rte_case14_redisp")
        # self.env_2_2 = make("rte_case14_redisp")
        # self.env_2_3 = make("rte_case14_redisp")
        # self.env_3_1 = make("rte_case14_redisp")
        # self.env_3_2 = make("rte_case14_redisp")
        # self.env_3_3 = make("rte_case14_redisp")

        self.my_agent_1_1 = DeepQAgent(self.env_1_1.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_1)
        # self.my_agent_1_2 = DeepQAgent(self.env_1_2.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_2)
        # self.my_agent_1_3 = DeepQAgent(self.env_1_3.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_2)
        # self.my_agent_2_1 = DeepQAgent(self.env_2_1.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_1)
        # self.my_agent_2_2 = DeepQAgent(self.env_2_2.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_1)
        # self.my_agent_2_3 = DeepQAgent(self.env_2_3.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_1)
        # self.my_agent_3_1 = DeepQAgent(self.env_3_1.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_2)
        # self.my_agent_3_2 = DeepQAgent(self.env_3_2.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_2)
        # self.my_agent_3_3 = DeepQAgent(self.env_3_3.action_space, mode="SAC", replace_action_space = True, new_int_action_space=self.new_int_action_space_2)
        self.trainer_1_1 = TrainAgent(level_flag='1_1', agent=self.my_agent_1_1, env=self.env_1_1, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_2)
        # self.trainer_1_2 = TrainAgent(level_flag='1_2', agent=self.my_agent_1_2, env=self.env_1_2, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_1)
        # self.trainer_1_3 = TrainAgent(level_flag='1_3', agent=self.my_agent_1_3, env=self.env_1_3, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_1)
        # self.trainer_2_1 = TrainAgent(level_flag='2_1', agent=self.my_agent_2_1, env=self.env_2_1, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_2)
        # self.trainer_2_2 = TrainAgent(level_flag='2_2', agent=self.my_agent_2_2, env=self.env_2_2, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_2)
        # self.trainer_2_3 = TrainAgent(level_flag='2_3', agent=self.my_agent_2_3, env=self.env_2_3, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_2)
        # self.trainer_3_1 = TrainAgent(level_flag='3_1', agent=self.my_agent_3_1, env=self.env_3_1, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_1)
        # self.trainer_3_2 = TrainAgent(level_flag='3_2', agent=self.my_agent_3_2, env=self.env_3_2, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_1)
        # self.trainer_3_3 = TrainAgent(level_flag='3_3', agent=self.my_agent_3_3, env=self.env_3_3, reward_fun=EconomicReward, another_int_action_space=self.new_int_action_space_1)

    def change_weights(self, agent, target_agent):
        W_agent, b_agent = agent.deep_q.model_policy.layers[7].get_weights()

        len_target = len(self.new_int_action_space_2)

        W_target = np.zeros((500, len_target))
        b_target = np.zeros(len_target)

        for i in range(len(self.new_int_action_space_1)):
            W_target[:, self.new_int_action_space_1[i]] = W_agent[:, i]
            b_target[self.new_int_action_space_1[i]] = b_agent[i]

        Weights_target = [W_target, b_target]
        target_agent.deep_q.model_policy.layers[0].set_weights(agent.deep_q.model_policy.layers[0].get_weights())
        target_agent.deep_q.model_policy.layers[1].set_weights(agent.deep_q.model_policy.layers[1].get_weights())
        target_agent.deep_q.model_policy.layers[2].set_weights(agent.deep_q.model_policy.layers[2].get_weights())
        target_agent.deep_q.model_policy.layers[3].set_weights(agent.deep_q.model_policy.layers[3].get_weights())
        target_agent.deep_q.model_policy.layers[4].set_weights(agent.deep_q.model_policy.layers[4].get_weights())
        target_agent.deep_q.model_policy.layers[5].set_weights(agent.deep_q.model_policy.layers[5].get_weights())
        target_agent.deep_q.model_policy.layers[6].set_weights(agent.deep_q.model_policy.layers[6].get_weights())
        target_agent.deep_q.model_policy.layers[7].set_weights(Weights_target)
        print("Transfer the policy successfully")


    def train(self):
        #pdb.set_trace()

        n_transfer = 0

        #self.trainer_2_2.run()
        self.trainer_1_1.run()
        print(self.trainer_1_1.evaluate_reward_list)
        #print(self.trainer_2_2.evaluate_reward_list)



        # self.trainer_1_1.start()
        # self.trainer_1_2.start()
        # self.trainer_1_3.start()
        # self.trainer_2_1.start()
        # self.trainer_2_2.start()
        # self.trainer_2_3.start()
        # self.trainer_3_1.start()
        # self.trainer_3_2.start()
        # self.trainer_3_3.start()

        # while self.trainer_1_1.train_not and self.trainer_1_2.train_not and self.trainer_1_3.train_not \
        # and self.trainer_2_1.train_not and self.trainer_2_2.train_not and self.trainer_2_3.train_not \
        # and self.trainer_3_1.train_not and self.trainer_3_2.train_not and self.trainer_3_3.train_not:
        #     if n_transfer > 21:
        #         break
        #     time.sleep(3600)
        #     if len(self.trainer_2_1.reward_episode) >= 10 and len(self.trainer_2_2.reward_episode) >= 10 and len(self.trainer_2_3.reward_episode) >= 10:
        #         self.reward_episode_record_2_1.append(self.trainer_2_1.reward_episode[-10:])
        #         self.reward_episode_record_2_2.append(self.trainer_2_2.reward_episode[-10:])
        #         self.reward_episode_record_2_3.append(self.trainer_2_3.reward_episode[-10:])
        #     if len(self.trainer_3_1.reward_episode) >= 10 and len(self.trainer_3_2.reward_episode) >= 10 and len(self.trainer_3_3.reward_episode) >= 10:
        #         self.reward_episode_record_3_1.append(self.trainer_3_1.reward_episode[-10:])
        #         self.reward_episode_record_3_2.append(self.trainer_3_2.reward_episode[-10:])
        #         self.reward_episode_record_3_3.append(self.trainer_3_3.reward_episode[-10:])
        #     if len(self.trainer_1_1.reward_episode) >= 10 and len(self.trainer_1_2.reward_episode) >= 10 and len(self.trainer_1_3.reward_episode) >= 10:
        #         self.reward_episode_record_1_1.append(self.trainer_1_1.reward_episode[-10:])
        #         self.reward_episode_record_1_2.append(self.trainer_1_2.reward_episode[-10:])
        #         self.reward_episode_record_1_3.append(self.trainer_1_3.reward_episode[-10:])
        #         if np.mean(self.trainer_1_2.reward_episode[-10:]) <= np.mean(self.trainer_1_3.reward_episode[-10:]):
        #             self.change_weights(self.my_agent_1_1, self.my_agent_1_2)
        #             self.transfer_record.append(1)
        #             print('1')
        #         else:
        #             self.change_weights(self.my_agent_1_1, self.my_agent_1_3)
        #             self.transfer_record.append(2)
        #             print('2')
        #             n_transfer += 1
        #     else:
        #         print('No transfer')

        # self.trainer_1_1.join()
        # self.trainer_1_2.join()
        # self.trainer_1_3.join()
        # self.trainer_2_1.join()
        # self.trainer_2_2.join()
        # self.trainer_2_3.join()
        # self.trainer_3_1.join()
        # self.trainer_3_2.join()
        # self.trainer_3_3.join()

        #print(self.trainer_1.reward_episode)
        #print(self.trainer_2_1.reward_episode)
        #print(self.trainer_2_2.reward_episode)

        print(self.trainer_1_1.evaluate_reward_list)
        # print(self.trainer_1_2.evaluate_reward_list)
        # print(self.trainer_1_3.evaluate_reward_list)
        # print(self.trainer_2_1.evaluate_reward_list)
        # print(self.trainer_2_2.evaluate_reward_list)
        # print(self.trainer_2_3.evaluate_reward_list)
        # print(self.trainer_3_1.evaluate_reward_list)
        # print(self.trainer_3_2.evaluate_reward_list)
        # print(self.trainer_3_3.evaluate_reward_list)

        print(self.reward_episode_record_1_1)
        # print(self.reward_episode_record_1_2)
        # print(self.reward_episode_record_1_3)
        # print(self.reward_episode_record_2_1)
        # print(self.reward_episode_record_2_2)
        # print(self.reward_episode_record_2_3)
        # print(self.reward_episode_record_3_1)
        # print(self.reward_episode_record_3_2)
        # print(self.reward_episode_record_3_3)


        print(self.transfer_record)


if __name__ == '__main__':
    env = grid2op.make("rte_case5_example", test=True)
    env.reset()
    DECAY_RATE = 0.9
    BUFFER_SIZE = 40000
    MINIBATCH_SIZE = 32
    TOT_FRAME = 3000000
    EPSILON_DECAY = 200000
    MIN_OBSERVATION = 64
    FINAL_EPSILON = 1/300  # have on average 1 random action per scenario of approx 287 time steps
    INITIAL_EPSILON = 0.3
    TAU = 0.01
    # Number of frames to "throw" into network
    NUM_FRAMES = 1 ## this has been changed compared to the original implementation.
    OBSERVATION_SIZE = env.observation_space.size()
    my_agent =  RandomAgent(env.action_space)
    NUM_ACTIONS = my_agent.action_space.n
    Time_Training = 600 # 88000

    ac = Action_classfication(env)
    ac.get_actions_levels()
    x = [ac.actions_level_1, ac.all_actions]

    ac = Action_classfication(env)
    ac.get_actions_levels()
    x = [ac.actions_level_1, ac.all_actions]

    replay_buffer_1_1 = ReplayBuffer(BUFFER_SIZE)
    replay_buffer_1_2 = ReplayBuffer(BUFFER_SIZE)
    replay_buffer_2 = ReplayBuffer(BUFFER_SIZE)
    replay_buffer_3 = ReplayBuffer(BUFFER_SIZE)

    replay_buffer_1_1.clear()
    replay_buffer_1_2.clear()
    replay_buffer_2.clear()
    replay_buffer_3.clear()

    test = MLRL_SAC_2(new_int_action_space = x)
    test.train()
