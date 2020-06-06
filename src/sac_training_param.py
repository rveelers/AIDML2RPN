""" This class is copied and edited from l2rpn_baselines.utils.TrainingParam """
import numpy as np


class TrainingParamSAC(object):
    """
    A class to store the training parameters of the models. It was hard coded in the getting_started/notebook 3
    of grid2op and put in this repository instead.
    """

    def __init__(self,
                 DECAY_RATE=0.9,
                 BUFFER_SIZE=40000,
                 MINIBATCH_SIZE=64,
                 STEP_FOR_FINAL_EPSILON=100000,  # step at which min_espilon is obtain
                 MIN_OBSERVATION=20,  # 5000  NOTE: the training does not start before min_observation steps....
                 FINAL_EPSILON=1./(7*288.),  # have on average 1 random action per week of approx 7*288 time steps
                 INITIAL_EPSILON=0.4,  # NOTE: epsilon is not really used in the updated version /Johan
                 TAU=0.01,
                 ALPHA=1,
                 NUM_FRAMES=1,
                 ALPHA_LR=3e-4,
                 AUTOMATIC_ALPHA_TUNING=True
                 ):

        self.DECAY_RATE = DECAY_RATE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.MIN_OBSERVATION = MIN_OBSERVATION  # 5000
        self.FINAL_EPSILON = float(FINAL_EPSILON)  # have on average 1 random action per day of approx 288 timesteps at the end (never kill completely the exploration)
        self.INITIAL_EPSILON = float(INITIAL_EPSILON)
        self.STEP_FOR_FINAL_EPSILON = float(STEP_FOR_FINAL_EPSILON)
        self.TAU = TAU
        self.NUM_FRAMES = NUM_FRAMES
        self.ALPHA = ALPHA
        self.ALPHA_LR = ALPHA_LR
        self.AUTOMATIC_ALPHA_TUNING = AUTOMATIC_ALPHA_TUNING

        self._exp_facto = np.log(self.INITIAL_EPSILON/self.FINAL_EPSILON)

    def get_next_epsilon(self, current_step):
        if current_step > self.STEP_FOR_FINAL_EPSILON:
            res = self.FINAL_EPSILON
        else:
            # exponential decrease
            res = self.INITIAL_EPSILON * np.exp(- (current_step / self.STEP_FOR_FINAL_EPSILON) * self._exp_facto )
        return res