# Credit Abhinav Sagar:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial
# Code under MIT license, available at:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial/blob/master/LICENSE

import numpy as np

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from deep_q_network import DeepQ
from hyper_parameters import BUFFER_SIZE
from replay_buffer import ReplayBuffer


class DeepQAgent(AgentWithConverter):
    # first change: An Agent must derived from grid2op.Agent (in this case MLAgent, because we manipulate vector instead
    # of classes)

    def convert_obs(self, observation):
        return observation.rho
        # return observation.to_vect()

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        # print("predict_movement_int: {}".format(predict_movement_int))
        return predict_movement_int

    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            # the first time an observation is observed, I set up the neural network with the proper dimensions.
            if self.mode == "DQN":
                cls = DeepQ
            else:
                raise RuntimeError("Unknown neural network named \"{}\"".format(self.mode))
            self.deep_q = cls(self.action_space.size(), observation_size=transformed_observation.shape[0], lr=self.lr)

    def __init__(self, action_space, mode="DQN", lr=1e-5):
        # this function has been adapted.

        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # compare to original implementation, i don't know the observation space size.
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.deep_q = None
        self.mode = mode
        self.lr = lr

    def load_network(self, path):
        # not modified compare to original implementation
        self.deep_q.load_network(path)

    def convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        # here i simply concatenate the action in case of multiple action in the "buffer"
        # this function existed in the original implementation, bus has been adapted.
        return np.concatenate(self.process_buffer)
