import os
import random
import time
import numpy as np

from copy import deepcopy
from progress_bar import print_progress

from grid2op import make
from grid2op.Action import TopologySetAction
from grid2op.Converter import IdToAct
from grid2op.Reward import L2RPNReward

path_grid = "l2rpn_2019"
env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)

run_id = 1
n = 10000
num_states = env.get_obs().rho.shape[0]
num_actions = env.action_space.size()
print('State space size:', num_states)
print('Action space size:', num_actions)

obs = env.reset()
converter = IdToAct(env.action_space)
converter.init_converter()
states = np.zeros((n, num_states))
rewards = np.zeros((n, num_actions))
cum_reward = 0.
reset_count = 0
start_time = time.time()

for i in range(n):
    print_progress(i+1, n, prefix='Sample {}/{}'.format(i+1, n), suffix='Episode count: {}'.format(reset_count))
    states[i] = env.get_obs().rho
    st = time.time()

    for act_id in range(env.action_space.size()):
        env_copy = deepcopy(env)
        act = converter.convert_act(act_id)
        obs, reward, done, _ = env_copy.step(act)
        rewards[i, act_id] = reward
        # print(act)
        # print(reward)

    # my_act = random.randrange(num_actions)
    my_act = np.argmax(rewards[i])
    obs, reward, done, _ = env.step(converter.convert_act(int(my_act)))

    # Reset environment when game over state is reached or max iterations is reached
    if done:
        reset_count += 1
        env.reset()

    cum_reward += reward
    if i % 1000 == 0:
        np.save(os.path.join('generated_data', 'states_{}_{}_{}.npy'.format(n, num_states, run_id)), states, allow_pickle=True)
        np.save(os.path.join('generated_data', 'rewards_{}_{}_{}.npy'.format(n, num_actions, run_id)), rewards, allow_pickle=True)

end_time = time.time() - start_time
# print(rewards)
print(cum_reward)
print('Time:', end_time)

np.save(os.path.join('generated_data', 'states_{}_{}_{}.npy'.format(n, num_states, run_id)), states, allow_pickle=True)
np.save(os.path.join('generated_data', 'rewards_{}_{}_{}.npy'.format(n, num_actions, run_id)), rewards, allow_pickle=True)
