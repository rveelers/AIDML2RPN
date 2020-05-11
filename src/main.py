import os
import grid2op
import matplotlib.pyplot as plt

from grid2op.Agent import DoNothingAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.Chronics import ChronicsHandler, Multifolder, GridStateFromFileWithForecasts
from grid2op.Environment import Environment
from grid2op.Parameters import Parameters
from grid2op.Plot import EpisodeReplay
from grid2op.Reward import RedispReward
from grid2op.Runner import Runner
from tqdm import tqdm

from deep_q_agent import DeepQAgent
from train_agent import TrainAgent

if __name__ == "__main__":
    power_grid_path = grid2op.CASE_14_FILE
    print(power_grid_path)
    multi_episode_path = grid2op.CHRONICS_MLUTIEPISODE
    max_iter = 100

    data_feeding = ChronicsHandler(chronicsClass=Multifolder,
                                   path=multi_episode_path,
                                   gridvalueClass=GridStateFromFileWithForecasts,
                                   max_iter=max_iter)
    backend = PandaPowerBackend()
    param = Parameters()
    env = Environment(init_grid_path=power_grid_path,
                      chronics_handler=data_feeding,
                      backend=backend,
                      parameters=param)

    my_agent = DeepQAgent(env.action_space, mode="DQN")
    trainer = TrainAgent(agent=my_agent, env=env, reward_fun=RedispReward)
    trainer.train(1000)
    trainer.agent.deep_q.save_network(os.path.join("..", "SavedNetworks", "saved_agent_" + trainer.agent.mode + ".h5"))

    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.deep_q.qvalue_evolution)
    plt.axhline(y=0, linewidth=3, color='red')
    plt.xlim(0, len(my_agent.deep_q.qvalue_evolution))
    plt.show()

    # Run the agent
    runner = Runner(**env.get_params_for_runner())
    path_agent = os.path.join("..", "Agents", "DeepQAgent")
    res = runner.run(nb_episode=1, max_iter=max_iter, path_save=path_agent, pbar=tqdm)

    # Print evaluation results
    print("The results for the trained agent are:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    # And now reload it and display the "movie" of this scenario
    plot_epi = EpisodeReplay(path_agent)
    plot_epi.replay_episode(res[0][1], max_fps=2, video_name="random_agent.gif")
