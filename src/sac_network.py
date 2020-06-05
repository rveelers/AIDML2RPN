import tensorflow_probability as tfp

from l2rpn_baselines.SAC.SAC_NN import SAC_NN
# from l2rpn_baselines.utils import TrainingParam

from sac_training_param import TrainingParamSAC


# TODO: how to pass training_params to this network?
# I want to use my "own" class TrainingParamSAC from sac_training_param. However, TrainingParam from
# l2rpn_baselines.utils seems to be used even though I pass TrainingParamSAC to the super()__init__() ??

# Specifically, when I call my_agent.train(...) on the SACAgent object (my_agent), the train function from
# DeepQAgent is called. On line 247 in DeepAgent, there is a check:
#      if training_step > max(training_param.MIN_OBSERVATION, training_param.MINIBATCH_SIZE):
# Since MIN_OBSERVATION is set to 5000 in the baselines, the training loop is not entered until step 5000 ...


class SACNetwork(SAC_NN):

    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 learning_rate_decay_steps=1000,
                 learning_rate_decay_rate=0.95,
                 training_param=TrainingParamSAC()):

        super().__init__(action_size,
                         observation_size,
                         lr=lr,
                         learning_rate_decay_steps=learning_rate_decay_steps,
                         learning_rate_decay_rate=learning_rate_decay_rate,
                         training_param=training_param)

    def predict_movement(self, data, epsilon, batch_size=None):
        """ Change (1): Deterministic --> stochastic policy """
        if batch_size is None:
            batch_size = data.shape[0]

        # Policy outputs a probability distribution over the actions
        p_actions = self.model_policy.predict(data, batch_size=batch_size)

        # create a distribution to sample from
        m = tfp.distributions.Categorical(probs=p_actions)

        # sample action from distribution
        action = m.sample()

        # Get probability for the chosen action
        prob = m.prob(action)

        return action, prob

    def summary(self):
        stringlist = []
        stringlist.append('model_value')
        self.model_value.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_Q')
        self.model_Q.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_policy')
        self.model_policy.summary(print_fn=lambda x: stringlist.append(x))

        short_model_summary = "\n".join(stringlist)
        return short_model_summary
