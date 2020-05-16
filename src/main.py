import os
import time
import grid2op
import matplotlib.pyplot as plt
from grid2op.Action.TopologySetAction import TopologySetAction

from grid2op.Plot import EpisodeReplay
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner

from deep_q_agent import DeepQAgent
from train_agent import TrainAgent


grid_paths = [
    "rte_case5_example",
    "rte_case14_test",
    "rte_case14_redisp",
    "rte_case14_realistic"
]


def get_network_file_name(grid, network_type, episodes, additional="", extension=".h5"):
    path_networks = "SavedNetworks"
    if not os.path.exists(path_networks):
        os.mkdir(path_networks)
    return os.path.join(path_networks, "agent_" + grid + "_" + network_type + "_" + str(episodes) + "_" + additional + extension)


def plot_grid_layout(save_file_path=None):
    plot_helper = PlotMatplot(env.observation_space)
    fig_layout = plot_helper.plot_layout()
    plt.show(fig_layout)
    if save_file_path is not None:
        plt.savefig(fname=save_file_path)


def plot_grid_observation(obs, save_file_path=None):
    plot_helper = PlotMatplot(env.observation_space)
    fig_layout = plot_helper.plot_obs(obs)
    if save_file_path is not None:
        plt.savefig(fname=save_file_path)
    plt.show(fig_layout)


def train_agent(agent, environment, num_episodes=10000, save_training_evaluation=True):
    trainer_agent = TrainAgent(agent=agent, env=environment, reward_fun=L2RPNReward)
    start = time.time()
    trainer_agent.train(num_episodes)
    print("Training time:  ", time.time() - start)
    trainer_agent.agent.deep_q.save_network(get_network_file_name(path_grid, trainer_agent.agent.mode, num_episodes))

    # Plot evaluation results
    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.deep_q.qvalue_evolution)
    plt.axhline(y=0, linewidth=3, color='red')
    plt.xlim(0, len(my_agent.deep_q.qvalue_evolution))
    if save_training_evaluation:
        plt.savefig(fname=get_network_file_name(path_grid, trainer_agent.agent.mode, num_episodes, additional="curve", extension=".png"))
    plt.show()
    return trainer_agent


def run_agent(environment, agent, max_iterations=100, plot_replay_episodes=True):
    runner = Runner(**environment.get_params_for_runner(), agentClass=None, agentInstance=agent)
    path_agents = "Agents"
    if not os.path.exists(path_agents):
        os.mkdir(path_agents)
    path_agents = os.path.join(path_agents, agent.__class__.__name__)
    res = runner.run(nb_episode=1, path_save=path_agents, max_iter=max_iterations)

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

    # Optional settings for running on GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turns running on GPU off
    # print(tf.config.list_physical_devices('GPU')) # Lists devices
    # tf.debugging.set_log_device_placement(True) # Logs which device is used per operation

    # Initialize the environment and agent
    path_grid = grid_paths[0]
    env = grid2op.make(path_grid, test=True, reward_class=L2RPNReward, action_class=TopologySetAction)
    my_agent = DeepQAgent(env.action_space, mode="DQN")

    # Plot grid visualization
    plot_grid_layout()

    # Load an existing network
    # my_agent.init_deep_q(my_agent.convert_obs(env.get_obs()))
    # my_agent.load_network(get_network_file_name(path_grid, "DQN", 10000))

    # Train a new network
    trainer = train_agent(my_agent, env, num_episodes=10000)

    # Run the agent
    run_agent(env, my_agent, max_iterations=100)

    # Plot final episode
    plot_grid_observation(env.get_obs())
