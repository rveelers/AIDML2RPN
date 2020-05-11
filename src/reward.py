import numpy as np
from grid2op.Environment import Environment

from grid2op.Reward import L2RPNReward
from grid2op.Reward import RedispReward


class L2RPNReward_LoadWise(L2RPNReward):
    """
    Update the L2RPN reward to take into account the fact that a change in the loads sum shall not be allocated as reward for the agent.

    """

    def __init__(self):
        super().__init__()

    def initialize(self, env):
        super().initialize(env)
        self.reward_min = - 10 * env.backend.n_line
        self.previous_loads = self.reward_max * np.ones(env.backend.n_line)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self._L2RPNReward__get_lines_capacity_usage(env)

            new_loads, _, _ = env.backend.loads_info()
            new_flows = np.abs(env.backend.get_line_flow())
            loads_variation = (np.sum(new_loads) - np.sum(self.previous_loads)) / np.sum(self.previous_loads)

            res = np.sum(line_cap + loads_variation)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res


class L2RPNReward_LoadWise_ActionWise(L2RPNReward):
    """
    Update the L2RPN reward to take into account the fact that a change in the loads sum shall not be allocated as reward for the agent.

    """

    def __init__(self):
        super().__init__()

    def initialize(self, env):
        super().initialize(env)
        self.reward_min = - 10 * env.backend.n_line
        self.previous_loads = self.reward_max * np.ones(env.backend.n_line)
        self.last_action = env.helper_action_env({})

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self._L2RPNReward__get_lines_capacity_usage(env)

            new_loads, _, _ = env.backend.loads_info()
            new_flows = np.abs(env.backend.get_line_flow())
            loads_variation = (np.sum(new_loads) - np.sum(self.previous_loads)) / np.sum(self.previous_loads)

            res = np.sum(line_cap + loads_variation)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min

        res -= (action != env.helper_action_env({})) * (action == self.last_action) * env.backend.n_line / 2

        self.last_action = action

        return res

