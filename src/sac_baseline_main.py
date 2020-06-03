import os
import time

from grid2op.Plot import EpisodeReplay
from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make

from l2rpn_baselines.SAC.SAC import SAC


def train_agent(agent, environment, network_path, num_iterations, logdir):
    start = time.time()
    agent.train(environment, num_iterations, network_path, logdir=logdir)
    print("Training time:  ", time.time() - start)
    # TODO: add plotting


def run_agent(environment, agent, num_iterations=100, plot_replay_episodes=True):
    runner = Runner(**environment.get_params_for_runner(), agentClass=None, agentInstance=agent)
    path_agents = "Agents"
    if not os.path.exists(path_agents):
        os.mkdir(path_agents)
    path_agents = os.path.join(path_agents, agent.__class__.__name__)
    res = runner.run(nb_episode=1, path_save=path_agents, max_iter=num_iterations)

    # Print run results and plot replay visualisation
    ep_replay = EpisodeReplay(agent_path=path_agents)
    print("The results for the trained agent are:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)
        if plot_replay_episodes:
            ep_replay.replay_episode(chron_name, video_name="episode.gif", display=False)

    return res


def main():
    NUM_TRAIN_ITERATIONS = 10000
    NUM_RUN_ITERATIONS = 10000

    # Initialize the environment and agent
    # path_grid = "l2rpn_2019"
    path_grid = 'rte_case14_realistic'
    # env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)
    env = make(path_grid)

    # Create SAC agent
    my_agent = SAC(action_space=env.action_space)
    network_path = os.path.join('saved_networks', 'Baselines', '{}_{}_{}'.format(path_grid, 'SAC', NUM_TRAIN_ITERATIONS))

    # Load baseline SAC network that has been trained for 10000 steps
    network_load_path = os.path.join(network_path, '_finished')
    my_agent.deep_q.load_network(network_load_path)

    # Train a new network
    # train_agent(my_agent, env, network_path, NUM_TRAIN_ITERATIONS, logdir=os.path.join(network_path, 'logs'))

    # Run the agent
    run_agent(env, my_agent, NUM_RUN_ITERATIONS, plot_replay_episodes=False)

if __name__ == "__main__":
    main()
