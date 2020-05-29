import numpy as np
from grid2op.Observation import CompleteObservation


class Observation:

    def __init__(self, observation: CompleteObservation):
        self.n = 2 * observation.n_line

    def __repr__(self):
        return np.concatenate
