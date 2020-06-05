import os

from grid2op.Plot import EpisodeReplay
from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make

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
    NUM_TRAIN_ITERATIONS = 1000
    NUM_RUN_ITERATIONS = 100
    path_grid = 'rte_case14_realistic'
    train_agent = True

    save_path = "saved_networks"
    logdir = 'logs'
    network_path = os.path.join(save_path, '{}_{}_{}'.format(path_grid, 'SACAgent', NUM_TRAIN_ITERATIONS))

    if not os.path.exists(network_path):
        os.mkdir(network_path)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Initialize the environment and agent
    env = make(path_grid, reward_class=L2RPNReward, action_class=TopologySetAction)
    my_agent = SACAgent(action_space=env.action_space)

    # Train the agent
    if train_agent:
        my_agent.train(env, NUM_TRAIN_ITERATIONS, network_path, logdir=logdir, training_param=TrainingParamSAC())
    else:
        obs = env.reset()
        transformed_obs = my_agent.convert_obs(obs)
        my_agent.init_deep_q(transformed_obs)
        # TODO: fix this ridiculous loading function
        my_agent.deep_q.load_network(os.path.join(network_path, my_agent.name))

    # Print summary of networks in SAC
    # print('\nSummary of networks in the SAC agent:\n', my_agent.summary())

    # Run the agent
    run_agent(env, my_agent, NUM_RUN_ITERATIONS, plot_replay_episodes=True)


if __name__ == "__main__":
    main()
