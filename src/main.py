import os
import time

import grid2viz
import matplotlib.pyplot as plt
from grid2op.Action import TopologyChangeAction

from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.Agent import DoNothingAgent
from grid2op.Plot import EpisodeReplay
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make

from deep_q_agent import DeepQAgent
from deep_q_network import DeepQ


# All available grids, if it gives an error remove the test=True flag in the make command
grid_paths = [
    "rte_case5_example",
    "rte_case14_test",
    "rte_case14_redisp",
    "rte_case14_realistic",
    "l2rpn_2019",  # The one we will use
    "l2rpn_case14_sandbox",
    "wcci_test"
]


def plot_grid_layout(environment, save_file_path=None):
    plot_helper = PlotMatplot(environment.observation_space)
    fig_layout = plot_helper.plot_layout()
    plt.show(fig_layout)
    if save_file_path is not None:
        plt.savefig(fname=save_file_path)


def plot_grid_observation(environment, observation=None, save_file_path=None):
    plot_helper = PlotMatplot(environment.observation_space)
    if observation is not None:
        fig_layout = plot_helper.plot_obs(observation)
    else:
        fig_layout = plot_helper.plot_obs(environment.get_obs())
    if save_file_path is not None:
        plt.savefig(fname=save_file_path)
    plt.show(fig_layout)


def train_agent(agent, environment, num_iterations=10000, imitation_learning=False):
    # Create folder for saving figures
    il = '_IL' if imitation_learning else ''
    network_path = os.path.join('saved_networks', '{}_{}_{}{}'.format(
        environment.name, agent.network_name, num_iterations, il))
    if not os.path.exists(network_path):
        os.mkdir(network_path)

    start = time.time()
    agent.train(environment, num_iterations, network_path)
    print("Training time:  ", time.time() - start)

    # Plot evaluation results
    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.reward_history)
    plt.savefig(fname=os.path.join(network_path, 'reward_history.png'))
    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.action_history)
    plt.savefig(fname=os.path.join(network_path, 'action_history.png'))
    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.deep_q.qvalue_evolution)
    plt.savefig(fname=os.path.join(network_path, 'q_value_evolution_curve.png'))
    plt.axhline(y=0, linewidth=3, color='red')
    plt.xlim(0, len(my_agent.deep_q.qvalue_evolution))
    plt.show()


def run_agent(environment, agent, num_iterations=100, plot_replay_episodes=True):
    agent.reset_action_history()
    agent.reset_action_history()
    runner = Runner(**environment.get_params_for_runner(), agentClass=None, agentInstance=agent)
    path_agents = "agents"
    if not os.path.exists(path_agents):
        os.mkdir(path_agents)
    path_agents = os.path.join(path_agents, agent.__class__.__name__)
    res = runner.run(nb_episode=1, path_save=path_agents, max_iter=num_iterations)

    plt.figure(figsize=(30, 20))
    plt.plot(agent.action_history)
    plt.show()
    plt.figure(figsize=(30, 20))
    plt.plot(agent.reward_history)
    plt.show()

    # Grid2Viz
    # import subprocess
    # subprocess.call(['grid2viz', '--agents_path', os.path.join('agents', 'DoNothingAgent'), '--env_path', environment.init_grid_path, '--port', str(8000)])

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


if __name__ == "__main__":

    # Initialize the environment and agent
    path_grid = "l2rpn_2019"
    # path_grid = grid_paths[0]
    env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)
    # env = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction)
    num_states = env.get_obs().rho.shape[0] + env.get_obs().line_status.shape[0]
    num_actions = env.action_space.size()
    print(num_states, num_actions)
    my_agent = DeepQAgent(env.action_space, num_states, network=DeepQ, env=env)
    # my_agent = DoNothingAgent(env.action_space)

    # Plot grid visualization
    # plot_grid_layout(env)

    if not os.path.exists('saved_networks'):
        os.mkdir('saved_networks')

    # Load an existing network
    # il_network_path = os.path.join('saved_networks', '{}_{}_{}_IL'.format(env.name, my_agent.network_name, 1000), 'network.h5')
    # il_network_path = os.path.join('saved_networks', '{}_{}_IL'.format(env.name, 10000), 'network.h5')
    # my_agent.load_network(il_network_path)

    # Train a new network
    train_agent(my_agent, env, num_iterations=1000, imitation_learning=False)

    # Run the agent
    run_agent(env, my_agent, num_iterations=100, plot_replay_episodes=True)

    # Plot final episode
    plot_grid_observation(env)
