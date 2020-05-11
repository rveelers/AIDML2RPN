import os
import grid2op
import matplotlib.pyplot as plt

from grid2op.Agent import DoNothingAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.Chronics import ChronicsHandler, Multifolder, GridStateFromFileWithForecasts
from grid2op.Environment import Environment
from grid2op.Parameters import Parameters
from grid2op.Reward import RedispReward
from grid2op.Runner import Runner
from tqdm import tqdm

from deep_q_agent import DeepQAgent
from train_agent import TrainAgent

if __name__ == "__main__":
    power_grid_path = grid2op.CASE_14_FILE
    print(power_grid_path)
    multi_episode_path = grid2op.CHRONICS_MLUTIEPISODE
    max_iter = 10

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
    trainer.train(100)
    trainer.agent.deep_q.save_network("saved_agent_" + trainer.agent.mode + ".h5")

    plt.figure(figsize=(30, 20))
    plt.plot(my_agent.deep_q.qvalue_evolution)
    plt.axhline(y=0, linewidth=3, color='red')
    plt.xlim(0, len(my_agent.deep_q.qvalue_evolution))
    plt.show()

    # OBSERVATION_SIZE = env.observation_space.size()
    # NUM_ACTIONS = my_agent.action_space.n

    # Run the agent
    # runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)
    # path_agent = os.path.join("..", "Agents", "DoNothingAgent")
    # res = runner.run(nb_episode=1, max_iter=max_iter, path_save=path_agent, pbar=tqdm)

    # # and now reload it and display the "movie" of this scenario
    # plot_epi = EpisodeReplay(path_agent)
    # plot_epi.replay_episode(res[0][1], max_fps=2, video_name="random_agent.gif")
