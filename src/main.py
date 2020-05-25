import os
import time
import matplotlib.pyplot as plt

from grid2op.Action.TopologySetAction import TopologySetAction
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


def train_agent(agent, environment, num_iterations=10000, save_training_evaluation=True):
    start = time.time()
    agent.train(environment, num_iterations)
    print("Training time:  ", time.time() - start)

    # Plot evaluation results
    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.deep_q.qvalue_evolution)
    plt.axhline(y=0, linewidth=3, color='red')
    plt.xlim(0, len(my_agent.deep_q.qvalue_evolution))
    if save_training_evaluation:
        network_path = os.path.join('saved_networks', 'agent_{}_{}_{}_curve.png'.format(environment.name, agent.network, num_iterations))
        plt.savefig(fname=network_path)
    plt.show()


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


if __name__ == "__main__":

    # Initialize the environment and agent
    path_grid = "l2rpn_2019"
    env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)
    num_states = env.get_obs().rho.shape[0]
    num_actions = env.action_space.size()
    my_agent = DeepQAgent(env.action_space, num_states, network=DeepQ)

    # Plot grid visualization
    # plot_grid_layout(env)

    # Load an existing network
    # network_path = os.path.join('saved_networks', 'agent_{}_{}_{}.h5'.format(env.name, my_agent.network, 10000))
    network_path = os.path.join('saved_networks', 'IL_{}_{}.h5'.format('l2rpn', 9000))
    my_agent.load_network(network_path)

    # Train a new network
    train_agent(my_agent, env, num_iterations=2000)

    # Run the agent
    run_agent(env, my_agent, num_iterations=5184)

    # Plot final episode
    plot_grid_observation(env)
