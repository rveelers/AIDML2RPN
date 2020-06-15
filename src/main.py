import os
import time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from grid2op.Action import TopologyChangeAction, TopologySetAction

from grid2op.Environment import Environment
from grid2op.Plot import EpisodeReplay
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make

from old_files.deep_q_agent import OldDeepQAgent
from l2rpn_baselines.DeepQSimple import DeepQSimple

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
    start = time.time()
    agent.train(environment, num_iterations, network_path)
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
        agent.reset_action_history()
        obs = environment.reset()
        reward = 0.
        cum_reward = 0.
        done = False
        iteration = 0
        log_path = os.path.join('logs', 'run', agent.id + '_' + str(time.time()))
        run_tf_writer = tf.summary.create_file_writer(log_path)
        for iteration in range(num_iterations):
            # act = agent.my_act(agent.convert_obs(obs), reward, done)
            predicted_qvalues = agent.deep_q.predict_rewards(agent.convert_obs(obs))
            sorted_qvalues = np.argsort(predicted_qvalues)[::-1]
            best_actions = sorted_qvalues[:4]
            best_action, best_reward = 0, 0.
            # _, best_reward, _, _ = obs.simulate(agent.convert_act(best_action))
            for action in best_actions:
                _, expected_reward, _, _ = obs.simulate(agent.convert_act(action))
                if expected_reward > best_reward:
                    best_action = action
                    best_reward = expected_reward

            print('In iteration', iteration, 'action', best_action, 'reward', reward)
            # print(agent.convert_act(act))
            # plot_grid_observation(environment)

            # Calculate predicted and expected rewards
            # expected_rewards = []
            # for action in range(agent.action_space.size()):
            #     _, expected_reward, _, _ = obs.simulate(agent.convert_act(action))
            #     expected_rewards.append(expected_reward)
            # sorted_rewards = np.argsort(expected_rewards)[::-1]

            obs, reward, done, _ = environment.step(agent.convert_act(best_action))
            cum_reward += reward

            # if len(agent.action_history) < 2 or agent.action_history[-2] != best_action:
                # print('In iteration', i, 'action', act, 'reward', reward)
                # print(agent.convert_act(best_action))

                # if not done:
                #     plot_grid_observation(environment)

                # Plot prediction vs expectation
            # plt.plot(predicted_qvalues)
            # plt.plot(np.array(expected_rewards) * (max(predicted_qvalues) / max(expected_rewards)))
            # plt.show()

            with run_tf_writer.as_default():
                tf.summary.scalar("action", best_action, iteration)
                tf.summary.scalar("reward", reward, iteration)
                tf.summary.scalar("cumulative reward", cum_reward, iteration)

            if done:
                break

        print(iteration, 'timesteps, reward:', cum_reward)


if __name__ == "__main__":
    # Initialize the environment and agent
    # path_grid = "rte_case5_example"
    path_grid = "rte_case14_redisp"
    env = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction)
    # env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)

    # my_agent = DeepQSimple(env.action_space, store_action=True)
    my_agent = OldDeepQAgent(env.action_space)

    num_states = my_agent.convert_obs(env.reset()).shape[0]
    num_actions = my_agent.action_space.size()
    num_training_iterations = 5000
    num_run_iterations = 5000

    print('State space size:', num_states)
    print('Action space size:', num_actions)
    print('Training iterations:', num_training_iterations)
    print('Run iterations:', num_run_iterations)

    # Plot grid visualization
    # plot_grid_layout(env)

    # Load an existing network
    # my_agent.id = '{}_{}_{}'.format(path_grid, my_agent.__class__.__name__, num_training_iterations)
    # my_agent.init_deep_q(my_agent.convert_obs(env.reset()))
    # my_agent.load(os.path.join('saved_networks', my_agent.id))

    # Load Imitation Learning network
    # num_samples = 1000
    # run_id = 0
    # il_network_path = '{}_{}_{}_IL'.format(path_grid, num_samples, run_id)
    # my_agent.init_deep_q(my_agent.convert_obs(env.reset()))
    # my_agent.load(os.path.join('saved_networks', il_network_path), name=il_network_path)

    # Train a new network
    my_agent.id = '{}_{}_{}'.format(path_grid, my_agent.__class__.__name__, num_training_iterations)
    train_agent(my_agent, env, num_iterations=num_training_iterations)

    # Run the agent
    path_grid = "rte_case14_realistic"
    env = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction)
    run_agent(env, my_agent, num_iterations=num_run_iterations, plot_replay_episodes=True, use_runner=False)
