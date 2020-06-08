import matplotlib.pyplot as plt
import numpy as np
from grid2op.Action import ActionSpace, SerializableActionSpace

from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import CompleteObservation
from grid2op.PlotGrid import PlotMatplot



class SplitAgent(BaseAgent):

    def __init__(self, action_space: SerializableActionSpace):
        BaseAgent.__init__(self, action_space)

        self.action_size = action_space.size()
        self.action_history = []
        self.reward_history = []
        self.step_count = 0

    def act(self, transformed_observation: CompleteObservation, reward, done=False, env: Environment=None):
        """ This method is called by the environment when using Runner. """
        # act = self.action_space({"set_bus": {"lines_ex_id": [(1, 2)], "lines_or_id": [(6, 2)]}, "set_line_status": [(2, -1)]})

        if self.step_count == 0:
            act = self.action_space({"change_bus": {"lines_or_id": [1, 2, 3], 'loads_id': [0]}})
        else:
            act = self.action_space({})

        self.action_history.append(act)
        self.reward_history.append(reward)
        self.step_count += 1

        return act

    def reset_action_history(self):
        self.action_history = []

    def reset_reward_history(self):
        self.reward_history = []
