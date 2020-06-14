import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from grid2op.Action import TopologyChangeAction
from grid2op.Plot import EpisodeReplay
from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.PlotGrid import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make

from sac_agent_baseline import SACBaselineAgent
from sac_agent import SACAgent
from sac_training_param import TrainingParamSAC


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
    NUM_TRAIN_ITERATIONS = 10001
    num_run_iterations = 1000
    path_grid = 'rte_case5_example'
    train_agent = True

    # Initialize the environment and agent
    environment = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction, test=True)  # TODO: why test?
    agent = SACAgent(action_space=environment.action_space)

    save_path = "saved_networks"
    logdir = os.path.join('logs2', agent.name + 'New', datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    rundir = os.path.join('runs', agent.name + 'New', datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))

    network_path = os.path.join(save_path, '{}_{}_{}'.format(path_grid, agent.name, NUM_TRAIN_ITERATIONS))

    if not os.path.exists(network_path):
        os.mkdir(network_path)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    # Train the agent
    if train_agent:
        agent.train(environment, NUM_TRAIN_ITERATIONS, network_path, logdir=logdir, training_param=TrainingParamSAC())
    else:
        # network_path = os.path.join(save_path, "rte_case5_example_SACAgent_80000_True_finished")  # TODO
        obs = environment.reset()
        transformed_obs = agent.convert_obs(obs)
        agent.init_deep_q(transformed_obs)
        agent.deep_q.load_network(network_path)

    # Print summary of networks in SAC
    # print('\nSummary of networks in the SAC agent:\n', my_agent.summary())

    # Run the agent
    # run_agent(env, my_agent, NUM_RUN_ITERATIONS, plot_replay_episodes=True)
    run_writer = tf.summary.create_file_writer(rundir, name=agent.name)
    obs = environment.reset()
    reward = 0.
    cum_reward = 0.
    act_old = None
    done = False
    for i in range(num_run_iterations):
        act = agent.my_act(agent.convert_obs(obs), reward, done)
        # _, reward, _, _ = obs.simulate(agent.convert_act(act))
        # _, reward_zero, _, _ = obs.simulate(agent.convert_act(0))
        # if reward_zero > reward:
        #    act = 0

        obs, reward, done, _ = environment.step(agent.convert_act(act))

        cum_reward += reward

        with run_writer.as_default():
            tf.summary.scalar("run/reward", reward, i)
            tf.summary.scalar("run/act", act, i)
            tf.summary.scalar("run/cum_reward", cum_reward, i)

        if act_old != act:
            print(i, act, reward, done)
            print(agent.convert_act(act))
            act_old = act

            #if not done:
            #    plot_helper = PlotMatplot(environment.observation_space)
            #    fig_layout = plot_helper.plot_obs(environment.get_obs())
            #    plt.show(fig_layout)

        if done:
            break

    print('Total reward: ', i, cum_reward)
    environment.reset()


if __name__ == "__main__":
    main()
