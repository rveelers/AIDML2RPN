from l2rpn_baselines.utils import DeepQAgent
from sac_network import SACNetwork


class SACAgent(DeepQAgent):

    # def __init__(self, action_space):
    #     super().__init__(action_space)

    def init_deep_q(self, transformed_observation):
        self.deep_q = SACNetwork(self.action_space.size(),
                                 observation_size=transformed_observation.shape[-1],
                                 lr=self.lr,
                                 learning_rate_decay_rate=self.learning_rate_decay_rate,
                                 learning_rate_decay_steps=self.learning_rate_decay_steps)

    def summary(self):
        return self.deep_q.summary()
