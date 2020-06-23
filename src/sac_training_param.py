"""Define TrainingParamSAC containing training parameters for SACAgent and SACNetwork."""
import numpy as np


class TrainingParamSAC(object):
    """"Training parameters in a class for convenience.

    Inspired by the L2RPN baseline repository: https://github.com/rte-france/l2rpn-baselines/
    """

    def __init__(self,
                 DECAY_RATE=0.90,
                 BUFFER_SIZE=40000,
                 MINIBATCH_SIZE=64,
                 STEP_FOR_FINAL_EPSILON=5000,
                 MIN_OBSERVATION=0,
                 FINAL_EPSILON=1/300,
                 INITIAL_EPSILON=0.5,
                 TAU=0.01,
                 lr=1e-5,
                 UPDATE_FREQ=100,  # Update tensorboard every UPDATE_FREQ steps.
                 SAVING_NUM=1000  # Save network every SAVING_NUM steps.
                 ):

        self.DECAY_RATE = DECAY_RATE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.MIN_OBSERVATION = MIN_OBSERVATION
        self.FINAL_EPSILON = float(FINAL_EPSILON)
        self.INITIAL_EPSILON = float(INITIAL_EPSILON)
        self.STEP_FOR_FINAL_EPSILON = float(STEP_FOR_FINAL_EPSILON)
        self.TAU = TAU
        self.lr = lr
        self.UPDATE_FREQ = UPDATE_FREQ
        self.SAVING_NUM = SAVING_NUM

        self._exp_facto = np.log(self.INITIAL_EPSILON/self.FINAL_EPSILON)

    def get_next_epsilon(self, current_step):
        if current_step > self.STEP_FOR_FINAL_EPSILON:
            res = self.FINAL_EPSILON
        else:
            # exponential decrease
            res = self.INITIAL_EPSILON * np.exp(- (current_step / self.STEP_FOR_FINAL_EPSILON) * self._exp_facto )
        return res