from os.path import join

from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.MakeEnv.Make import make

from l2rpn_baselines.SAC import train, evaluate

save_path = "saved_networks/Baselines/SAC"
load_path = save_path
logs_dir = join(save_path, 'logs')

# path_grid = "l2rpn_2019" TODO: running on this environment gives weird errors
# env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)

env = make()  # runs on rte_case14_realistic
res1 = train(env, logs_dir=logs_dir, save_path=save_path, iterations=100)
res2 = evaluate(env, load_path=load_path, nb_episode=1)  # TODO: why is the progress bar broken?

