import os
import grid2op
import matplotlib.pyplot as plt

from grid2op.Plot import EpisodeReplay
from grid2op.PlotGrid.PlotMatplot import PlotMatplot
from grid2op.Reward import RedispReward
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner

from deep_q_agent import DeepQAgent
from train_agent import TrainAgent

if __name__ == "__main__":

    # Initialize the environment and agent
    env = grid2op.make("rte_case14_redisp", test=True, reward_class=L2RPNReward)
    my_agent = DeepQAgent(env.action_space, mode="DQN")

    # Load an existing network
    # my_agent.init_deep_q(my_agent.convert_obs(env.get_obs()))
    # my_agent.load_network(os.path.join("SavedNetworks", "saved_agent_" + "DQN" + ".h5"))

    # Train a new network
    trainer = TrainAgent(agent=my_agent, env=env, reward_fun=RedispReward)
    trainer.train(1000)
    path_networks = "SavedNetworks"
    if not os.path.exists(path_networks):
        os.mkdir(path_networks)
    trainer.agent.deep_q.save_network(os.path.join(path_networks, "saved_agent_" + trainer.agent.mode + ".h5"))

    # Plot evaluation results
    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.deep_q.qvalue_evolution)
    plt.axhline(y=0, linewidth=3, color='red')
    plt.xlim(0, len(my_agent.deep_q.qvalue_evolution))
    plt.show()

    # Run the agent
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
    path_agents = "Agents"
    max_iter = 30  # to save time we only assess performance on 30 iterations
    if not os.path.exists(path_agents):
        os.mkdir(path_agents)
    path_agents = os.path.join(path_agents, "DeepQAgent")
    res = runner.run(nb_episode=1, path_save=path_agents, max_iter=max_iter)

    # Print run results
    print("The results for the trained agent are:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    # Plot grid visualization
    plot_helper = PlotMatplot(env.observation_space)
    # fig_layout = plot_helper.plot_layout()
    # fig_layout = plot_helper.plot_info(line_values=env._thermal_limit_a)
    fig_layout = plot_helper.plot_obs(env.get_obs())
    plt.show(fig_layout)

    # Plot run visualisation
    ep_replay = EpisodeReplay(agent_path=path_agents)
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        ep_replay.replay_episode(chron_name, video_name="episode.gif", display=False)
