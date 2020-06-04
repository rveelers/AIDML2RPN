from l2rpn_baselines.SAC import SAC_NN


class SACNetwork(SAC_NN):

    def __init__(self, action_size, observation_size):
        super().__init__(self, action_size, observation_size)

    def summary(self):
        stringlist = []
        stringlist.append('model_value_target')
        self.model_value_target.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_value')
        self.model_value.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_Q')
        self.model_Q.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_Q2')
        self.model_Q2.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_policy')
        self.model_policy.summary(print_fn=lambda x: stringlist.append(x))

        short_model_summary = "\n".join(stringlist)
        return short_model_summary
