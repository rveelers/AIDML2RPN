import os
import numpy as np
import matplotlib.pyplot as plt

from deep_q_network import DeepQ
from progress_bar import print_progress


def plot_state_prediction(state, reward):
    Y = nn.predict_rewards(state)
    X = np.arange(0, reward.shape[0], step=1)
    Y_expected = reward
    plt.plot(X, Y, label='predicted')
    plt.plot(X, Y_expected, label='expected')
    plt.legend()
    plt.show()


BATCH_SIZE = 8
NUM_EPISODES = 100
run_id = 0
n = 3000
num_states = 20
num_actions = 76
iterations = n // BATCH_SIZE + 1
nn = DeepQ(num_actions, num_states)

states = np.load(os.path.join('generated_data', 'states_{}_{}_{}.npy'.format(n, num_states, 0)), allow_pickle=True)
rewards = np.load(os.path.join('generated_data', 'rewards_{}_{}_{}.npy'.format(n, num_actions, 0)), allow_pickle=True)

loss = 0
for episode in range(NUM_EPISODES):
    print_progress(episode, NUM_EPISODES, prefix='Episode {}/{}'.format(episode, NUM_EPISODES))
    for iteration in range(iterations):
        start_index = iteration * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        state_batch = states[start_index:end_index]
        target_batch = rewards[start_index:end_index]
        loss = nn.train_imitation(state_batch, target_batch)

print("We had an imitation loss equal to ", loss)
path = os.path.join('SavedNetworks', 'IL_l2rpn_{}.h5'.format(n))
nn.save_network(path)

# nn.load_network(path)
plot_state_prediction(states[10], rewards[10])
