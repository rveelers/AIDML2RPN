DISCOUNT_RATE = 0.9
LEARNING_RATE = 1e-5
BUFFER_SIZE = 10000
BATCH_SIZE = 64
TRAIN_INTERVAL = 5
REPLACE_TARGET_INTERVAL = 256
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
TAU = 0.01
ALPHA = 1
NUM_FRAMES = 1

# Hyperparameters from grid2op getting_started
# DECAY_RATE = 0.9
# BUFFER_SIZE = 40000
# MINIBATCH_SIZE = 64
# TOT_FRAME = 3000000
# EPSILON_DECAY = 10000
# MIN_OBSERVATION = 42 #5000
# FINAL_EPSILON = 1/300  # have on average 1 random action per scenario of approx 287 time steps
# INITIAL_EPSILON = 0.1
# TAU = 0.01
# ALPHA = 1
# # Number of frames to "throw" into network
# NUM_FRAMES = 1 ## this has been changed compared to the original implementation.
