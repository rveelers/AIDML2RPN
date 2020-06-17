import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from grid2op.Action import TopologyChangeAction
from grid2op.Plot import EpisodeReplay
from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.PlotGrid import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Runner import Runner
from grid2op.MakeEnv.Make import make
from grid2op.Agent import DoNothingAgent

from l2rpn_baselines_old.SAC.SAC import SAC as SACBaselineAgent

# from sac_agent_baseline import SACBaselineAgent
from sac_agent import SACAgent
from sac_training_param import TrainingParamSAC


def run_baseline_sac(num_run_iterations):
    environment = make('rte_case14_realistic', reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
    agent = SACBaselineAgent(action_space=environment.action_space,
                             name="SACBaselineAgent",
                             lr=1e-5,
                             learning_rate_decay_steps=5000,
                             learning_rate_decay_rate=1,  # TODO correct?
                             store_action=False,
                             istraining=False)

    obs = environment.reset()
    transformed_obs = agent.convert_obs(obs)
    agent.init_deep_q(transformed_obs)

    rundir = os.path.join('runs', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    network_path = r'C:\Users\johan\Documents\GitHub\AIDML2RPN\src\saved_networks' \
                   r'\rte_case14_redisp_SACBaselineAgent_5001'
    agent.load(network_path)

    # Run the agent
    run_writer = tf.summary.create_file_writer(rundir, name=agent.name)
    obs = environment.reset()
    cum_reward = 0.
    act_old = None

    for i in range(num_run_iterations):
        act = agent.my_act(agent.convert_obs(obs), reward=0)  # TODO why need reward here?

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


def load_sac(network_path):
    environment = make('rte_case14_realistic', reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
    agent = SACAgent(action_space=environment.action_space, name='SACAgentNew')
    obs = environment.reset()
    transformed_obs = agent.convert_obs(obs)
    agent.init_deep_q(transformed_obs)
    agent.deep_q.load_network(network_path)
    return environment, agent


def run_do_nothing(num_run_iterations):
    environment = make('rte_case14_realistic', reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
    agent = DoNothingAgent(environment.action_space)

    rundir = os.path.join('runs', 'DoNothingAgent', datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    # Run the agent
    run_writer = tf.summary.create_file_writer(rundir, name='DoNothingAgent')
    obs = environment.reset()

    cum_reward = 0.

    tot_load = []
    tot_prod = []
    for i in range(num_run_iterations):
        act = agent.act(obs, reward=0)

        obs, reward, done, _ = environment.step(act)

        tot_load.append(sum(obs.load_p))
        tot_prod.append(sum(obs.prod_p))

        print('i:', i, '\tact:', 0, '\treward:', reward, '\ttot load', tot_load[i], '\ttot prod', tot_prod[i])
        cum_reward += reward

        with run_writer.as_default():
            tf.summary.scalar("run/reward", reward, i)
            tf.summary.scalar("run/act", 0, i)
            tf.summary.scalar("run/cum_reward", cum_reward, i)

        if done:
            break

    print('Total reward: ', i, cum_reward)

    plt.plot(tot_prod)
    plt.plot(tot_load)
    plt.legend(['Total production', 'Total load'])
    plt.xlabel('timestep')
    plt.ylabel('Power [MW]')

    plt.show()

    environment.reset()


def run_sac_new(agent, num_run_iterations):
    environment = make('rte_case14_realistic', reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)

    rundir = os.path.join('runs', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    # Run the agent
    run_writer = tf.summary.create_file_writer(rundir, name=agent.name)
    obs = environment.reset()
    cum_reward = 0.
    act_old = None

    proposed_acts = []
    tot_load = []
    tot_prod = []
    for i in range(num_run_iterations):
        acts, probs = agent.my_acts(agent.convert_obs(obs), nr_acts=1)
        proposed_acts.append(acts)

        # Simulate 0-action and the top 4 proposed actions and pick the top one
        #act = 0
        #_, rw_to_beat, _, _ = obs.simulate(agent.convert_act(act))
        # rw_to_beat = -100000
        # for act_proposal in acts:
        #    _, rw_proposal, _, _ = obs.simulate(agent.convert_act(act_proposal))
        #    if rw_proposal > rw_to_beat:
        #        act = act_proposal
        #        rw_to_beat = rw_proposal

        act = acts[0]

        if act != act_old:
            print(agent.convert_act(act))
            act_old = act

        obs, reward, done, _ = environment.step(agent.convert_act(act))
        tot_load.append(sum(obs.load_p))
        tot_prod.append(sum(obs.prod_p))

        print('i:', i, '\tact:', act, '\treward:', reward, '\tproposed acts:', acts, tot_load[i], '\ttot prod', tot_prod[i])
        cum_reward += reward

        with run_writer.as_default():
            tf.summary.scalar("run/reward", reward, i)
            tf.summary.scalar("run/act", act, i)
            tf.summary.scalar("run/cum_reward", cum_reward, i)

        if done:
            break

    print('Total reward: ', i, cum_reward)
    environment.reset()

    plt.plot(tot_prod)
    plt.plot(tot_load)
    plt.legend(['Total production', 'Total load'])
    plt.xlabel('timestep')
    plt.ylabel('Power [MW]')

    plt.show()

    proposed_acts_np = np.array(proposed_acts).flatten()
    plt.hist(proposed_acts_np)
    plt.show()

    for act in np.unique(proposed_acts_np):
        print(act, agent.convert_act(act))

def main():
    NUM_TRAIN_ITERATIONS = 5006
    NUM_RUN_ITERATIONS = 5000
    path_grid = 'rte_case14_redisp'
    save_path = "saved_networks"

    train_agent = True  # OBS!!

    run_agent = False

    if train_agent:
        # Initialize the environment and agent
        environment = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
        agent = SACAgent(action_space=environment.action_space, name='SACAgentNew')

        logdir = os.path.join('logs2', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
        network_path = os.path.join(save_path, '{}_{}_{}'.format(path_grid, agent.name, NUM_TRAIN_ITERATIONS))

        for path in [logdir, network_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        agent.train(environment, NUM_TRAIN_ITERATIONS, network_path, logdir=logdir, training_param=TrainingParamSAC())

    if run_agent:
        run_do_nothing(NUM_RUN_ITERATIONS)

        # environment, agent = load_sac(r'C:\Users\johan\Documents\GitHub\AIDML2RPN\src\saved_networks\rte_case14_redisp_SACAgent_50000')
        # run_sac_new(agent, NUM_RUN_ITERATIONS)
        # plot_grid_layout(environment)

        #run_baseline(NUM_RUN_ITERATIONS)


def plot_grid_layout(environment, save_file_path=None):
    plot_helper = PlotMatplot(environment.observation_space)
    fig_layout = plot_helper.plot_layout()
    plt.show(fig_layout)
    if save_file_path is not None:
        plt.savefig(fname=save_file_path)


if __name__ == "__main__":
    main()
