import os
import numpy as np
import matplotlib.pyplot as plt
from l2rpn_baselines.DeepQSimple.DeepQ_NN import DeepQ_NN

from progress_bar import print_progress


def plot_state_prediction(state, reward):
    Y = nn.predict_rewards(state)
    X = np.arange(0, reward.shape[0], step=1)
    Y_expected = reward
    plt.plot(X, Y, label='predicted')
    plt.plot(X, Y_expected, label='expected')
    plt.xlabel('Action ID')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()


IL_LEARNING_RATE = 1e-3
IL_BATCH_SIZE = 8
NUM_EPISODES = 100
run_id = 1
n = 1000
num_states = 166
num_actions = 76
iterations = n // IL_BATCH_SIZE + 1
# nn = DeepQ(num_actions, num_states, lr=IL_LEARNING_RATE)
nn = DeepQ_NN(num_actions, num_states, lr=IL_LEARNING_RATE)

states = np.load(os.path.join('generated_data', 'states_{}_{}_{}.npy'.format(n, num_states, run_id)), allow_pickle=True)
rewards = np.load(os.path.join('generated_data', 'rewards_{}_{}_{}.npy'.format(n, num_actions, run_id)), allow_pickle=True)

loss = 0
for episode in range(NUM_EPISODES):
    print_progress(episode, NUM_EPISODES, prefix='Episode {}/{}'.format(episode, NUM_EPISODES))
    for iteration in range(iterations):
        start_index = iteration * IL_BATCH_SIZE
        end_index = start_index + IL_BATCH_SIZE
        state_batch = states[start_index:end_index]
        target_batch = rewards[start_index:end_index]
        loss = nn.train_imitation(state_batch, target_batch)

print("We had an imitation loss equal to ", loss)
network_path = os.path.join('saved_networks', 'IL_{}_{}_{}'.format('l2rpn_2019', n, run_id))
if not os.path.exists(network_path):
    os.mkdir(network_path)
nn.save_network(network_path)

# nn.load_network(path)
plot_state_prediction(states[10], rewards[10])
