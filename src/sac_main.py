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

from l2rpn_baselines_old.SAC.SAC import SAC as SACBaselineAgent

# from sac_agent_baseline import SACBaselineAgent
from sac_agent import SACAgent
from sac_training_param import TrainingParamSAC


def run_agent(agent, num_run_iterations, rundir):
    environment = make('rte_case14_realistic', reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
    # Run the agent
    # run_agent(env, my_agent, NUM_RUN_ITERATIONS, plot_replay_episodes=True)
    run_writer = tf.summary.create_file_writer(rundir, name=agent.name)
    obs = environment.reset()
    cum_reward = 0.
    act_old = None
    for i in range(num_run_iterations):
        acts, probs = agent.my_acts(agent.convert_obs(obs), nr_acts=4)

        # Simulate 0-action and the top 4 proposed actions and pick the top one
        act = 0
        _, rw_to_beat, _, _ = obs.simulate(agent.convert_act(act))
        for act_proposal in acts:
            _, rw_proposal, _, _ = obs.simulate(agent.convert_act(act_proposal))
            if rw_proposal > rw_to_beat:
                act = act_proposal
                rw_to_beat = rw_proposal

        if act != act_old:
            print(agent.convert_act(act))
            act_old = act

        obs, reward, done, _ = environment.step(agent.convert_act(act))
        print('i:', i, '\tact:', act, '\treward:', reward)
        cum_reward += reward

        with run_writer.as_default():
            tf.summary.scalar("run/reward", reward, i)
            tf.summary.scalar("run/act", act, i)
            tf.summary.scalar("run/cum_reward", cum_reward, i)

        if done:
            break

    print('Total reward: ', i, cum_reward)
    environment.reset()


def main():
    NUM_TRAIN_ITERATIONS = 5001
    NUM_RUN_ITERATIONS = 5000
    path_grid = 'rte_case14_redisp'
    train_agent = True
    imitation_learning = False

    # Initialize the environment and agent
    environment = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
    # agent = SACAgent(action_space=environment.action_space, name='SACAgentNew')

    agent = SACBaselineAgent(action_space=environment.action_space,
                             name="SACBaselineAgent",
                             lr=1e-5,
                             learning_rate_decay_steps=5000,
                             learning_rate_decay_rate=1,
                             store_action=False,
                             istraining=False)

    save_path = "saved_networks"
    logdir = os.path.join('logs2', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    rundir = os.path.join('runs', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    network_path = os.path.join(save_path, '{}_{}_{}'.format(path_grid, agent.name, NUM_TRAIN_ITERATIONS))

    for path in [logdir, rundir, network_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Train the agent
    if train_agent:
        agent.train(environment, NUM_TRAIN_ITERATIONS, network_path, logdir=logdir, training_param=TrainingParamSAC())
    else:
        obs = environment.reset()
        transformed_obs = agent.convert_obs(obs)
        agent.init_deep_q(transformed_obs)
        agent.deep_q.load_network(network_path)

    # Run the agent
    # run_agent(agent, NUM_RUN_ITERATIONS, rundir)


if __name__ == "__main__":
    main()
