import os
import time
import matplotlib.pyplot as plt
from grid2op.Action import TopologyChangeAction

from grid2op.Environment import Environment
from grid2op.Plot import EpisodeReplay
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make

# All available grids, if it gives an error remove the test=True flag in the make command
from deep_q_agent import DeepQAgent
from split_agent import SplitAgent

grid_paths = [
    "rte_case5_example",
    "rte_case14_test",
    "rte_case14_redisp",
    "rte_case14_realistic",
    "l2rpn_2019",  # The one we will use
    "l2rpn_case14_sandbox",
    "wcci_test"
]


def plot_grid_layout(environment: Environment, save_file_path=None):
    plot_helper = PlotMatplot(environment.observation_space)
    fig_layout = plot_helper.plot_info(line_values=environment.name_line)
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


def train_agent(agent, environment, num_iterations):
    if not os.path.exists('saved_networks'):
        os.mkdir('saved_networks')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    network_path = os.path.join('saved_networks', agent.id)
    log_path = os.path.join('logs', agent.id)
    start = time.time()
    agent.train(environment, num_iterations, network_path, log_path)
    print("Training time:  ", time.time() - start)


def run_agent(environment, agent, num_iterations, plot_replay_episodes=True, use_runner=True):
    if use_runner:
        if not os.path.exists('agents'):
            os.mkdir('agents')

        runner = Runner(**environment.get_params_for_runner(), agentClass=None, agentInstance=agent)
        path_agents = os.path.join('agents', agent.id)
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

    else:
        obs = environment.reset()
        reward = 0.
        cum_reward = 0.
        done = False
        for i in range(num_iterations):
            # plot_grid_observation(environment)
            act = agent.my_act(agent.convert_obs(obs), reward, done)
            # act = agent.act(obs, reward, done)
            obs, reward, done, _ = environment.step(agent.convert_act(act))
            # obs, reward, done, _ = environment.step(act)
            cum_reward += reward

            if done:
                break

        print(i, 'timesteps, reward:', cum_reward)


if __name__ == "__main__":
    # Initialize the environment and agent
    # path_grid = "l2rpn_2019"
    path_grid = "rte_case5_example"
    # path_grid = "rte_case14_realistic"
    # env = make(path_grid, test=True, reward_class=L2RPNReward, action_class=TopologySetAction)
    env = make(path_grid, test=True, reward_class=L2RPNReward, action_class=TopologyChangeAction)
    # env = make(path_grid, reward_class=L2RPNReward)

    # my_agent = DoNothingAgent(env.action_space)
    # my_agent = DoubleDuelingDQN(num_states, env.action_space, is_training=True)
    my_agent = DeepQAgent(env.action_space, store_action=True)
    # my_agent = SAC(env.action_space, store_action=True)
    # my_agent = SplitAgent(env.action_space)

    # num_states = my_agent.convert_obs(env.reset()).shape[1]
    num_actions = env.action_space.size()
    num_training_iterations = 100000
    num_run_iterations = 1000

    # print('State space size:', num_states)
    print('Action space size:', num_actions)
    print('Training iterations:', num_training_iterations)
    print('Run iterations:', num_run_iterations)

    # Plot grid visualization
    # plot_grid_layout(env)

    # Load an existing network
    # my_agent.id = '{}_{}_{}'.format(path_grid, my_agent.__class__.__name__, num_training_iterations)
    # agent_id = 'IL_{}_{}_{}'.format(path_grid, my_agent.__class__.__name__, num_training_iterations)
    # my_agent.init_deep_q(my_agent.convert_obs(env.reset()))
    # my_agent.load(os.path.join('saved_networks', my_agent.id))

    # Train a new network
    my_agent.id = '{}_{}_{}'.format(path_grid, my_agent.__class__.__name__, num_training_iterations)
    train_agent(my_agent, env, num_iterations=num_training_iterations)

    # Run the agent
    # run_agent(env, my_agent, num_iterations=num_run_iterations, plot_replay_episodes=True, use_runner=False)
