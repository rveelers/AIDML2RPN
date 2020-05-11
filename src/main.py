import os
import grid2op

from grid2op.Agent import DoNothingAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.Chronics import ChronicsHandler, Multifolder, GridStateFromFileWithForecasts
from grid2op.Environment import Environment
from grid2op.Parameters import Parameters
from grid2op.Plot import EpisodeReplay
from grid2op.Runner import Runner
from tqdm import tqdm

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

    runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)
    path_agent = os.path.join("..", "Agents", "DoNothingAgent")
    res = runner.run(nb_episode=1, max_iter=max_iter, path_save=path_agent, pbar=tqdm)

    # and now reload it and display the "movie" of this scenario
    plot_epi = EpisodeReplay(path_agent)
    # plot_epi.replay_episode(res[0][1], max_fps=2, video_name="random_agent.gif")
