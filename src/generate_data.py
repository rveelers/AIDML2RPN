import os
import time
import numpy as np

from progress_bar import print_progress

from grid2op.Action import TopologyChangeAction
from grid2op.MakeEnv.Make import make
from grid2op.Converter import IdToAct
from grid2op.Reward import L2RPNReward


def convert_obs(observation):
    """ Convert a CompleteObservation object from Grid2Op to a vector as input for the Q-network.
    This method equals the convert_obs method in DeepQAgent. """
    return np.concatenate((
        observation.prod_p / 150,
        observation.load_p / 120,
        observation.rho / 2,
        observation.timestep_overflow / 10,
        observation.line_status,
        (observation.topo_vect + 1) / 3,
        observation.time_before_cooldown_line / 10,
        observation.time_before_cooldown_sub / 10))


# Setup the environment
path_grid = "rte_case14_redisp"
env = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction)
obs = env.reset()

run_id = 0
n = 1000
num_states = convert_obs(obs).shape[0]
num_actions = 191  # Specific for TopologyChangeAction on case 14
print('State space size:', num_states)
print('Action space size:', num_actions)

converter = IdToAct(env.action_space)
converter.init_converter()
states = np.zeros((n, num_states))
rewards = np.zeros((n, num_actions))
cum_reward = 0.
reset_count = 0
start_time = time.time()

# Generate n samples ...
for i in range(n):
    print_progress(i+1, n, prefix='Sample {}/{}'.format(i+1, n), suffix='Episode count: {}'.format(reset_count))
    states[i] = convert_obs(obs)
    st = time.time()

    # ... by simulating all actions and storing the rewards
    for act_id in range(num_actions):
        act = converter.convert_act(act_id)
        _, reward, _, _ = obs.simulate(act)
        rewards[i, act_id] = reward

    my_act = np.argmax(rewards[i])
    obs, reward, done, _ = env.step(converter.convert_act(int(my_act)))

    # Reset environment when game over state is reached
    if done:
        reset_count += 1
        env.reset()

    cum_reward += reward
    if i % 100 == 0:
        np.save(os.path.join('generated_data', 'states_{}_{}_{}_{}.npy'.format(path_grid, n, num_states, run_id)), states, allow_pickle=True)
        np.save(os.path.join('generated_data', 'rewards_{}_{}_{}_{}.npy'.format(path_grid, n, num_actions, run_id)), rewards, allow_pickle=True)

end_time = time.time() - start_time
print(cum_reward)
print('Time:', end_time)

# Save the state vectors and the corresponding reward vectors in a numpy file
np.save(os.path.join('generated_data', 'states_{}_{}_{}_{}.npy'.format(path_grid, n, num_states, run_id)), states, allow_pickle=True)
np.save(os.path.join('generated_data', 'rewards_{}_{}_{}_{}.npy'.format(path_grid, n, num_actions, run_id)), rewards, allow_pickle=True)
