import os
import numpy as np
import matplotlib.pyplot as plt

from deep_q_network import DeepQ
from progress_bar import print_progress

path_grid = "rte_case14_redisp"
IL_LEARNING_RATE = 1e-3
IL_BATCH_SIZE = 8
NUM_EPISODES = 100
run_id = 0
n = 1000
num_states = 166
num_actions = 191
iterations = n // IL_BATCH_SIZE + 1
nn = DeepQ(num_actions, num_states, lr=IL_LEARNING_RATE)

# Load the samples we want to train on
states = np.load(os.path.join('generated_data', 'states_{}_{}_{}_{}.npy'.format(path_grid, n, num_states, run_id)), allow_pickle=True)
rewards = np.load(os.path.join('generated_data', 'rewards_{}_{}_{}_{}.npy'.format(path_grid, n, num_actions, run_id)), allow_pickle=True)

loss = 0
for episode in range(NUM_EPISODES):
    print_progress(episode, NUM_EPISODES, prefix='Episode {}/{}'.format(episode, NUM_EPISODES))

    # Train the network on batches of the generated samples
    for iteration in range(iterations):
        start_index = iteration * IL_BATCH_SIZE
        end_index = start_index + IL_BATCH_SIZE
        state_batch = states[start_index:end_index]
        target_batch = rewards[start_index:end_index]
        loss = nn.model.train_on_batch(state_batch, target_batch)

# print("We had an imitation loss equal to ", loss)
network_path = os.path.join('saved_networks', 'imitation_learning', '{}_{}_{}_il'.format(path_grid, n, run_id))
nn.save_network(network_path)

# Plot single sample for visualization
nn.load_network(network_path)
state = states[10]
reward = rewards[10]
Y = nn.model.predict(state.reshape(1, num_states), batch_size=1)[0]
X = np.arange(0, reward.shape[0], step=1)
Y_expected = reward
plt.plot(X, Y_expected, label='expected', alpha=0.8)
plt.plot(X, Y, label='predicted', alpha=0.8)
plt.xlabel('Action ID')
plt.ylabel('Reward')
plt.legend(loc='upper left')
plt.show()
