""" Main file for training and running SAC Agents."""
import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from grid2op.Action import TopologyChangeAction
from grid2op.PlotGrid import PlotMatplot
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.MakeEnv.Make import make
from grid2op.Agent import DoNothingAgent

from sac_agent import SACAgent
from sac_training_param import TrainingParamSAC


def run_do_nothing(environment, num_run_iterations):
    """ Run the DoNothingAgent and print/plot/write statistics."""
    agent = DoNothingAgent(environment.action_space)

    # Create Tensorboard writer
    rundir = os.path.join('runs', 'DoNothingAgent', datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    run_writer = tf.summary.create_file_writer(rundir, name='DoNothingAgent')

    # Run the agent
    obs = environment.reset()
    cum_reward = 0.
    tot_load = []
    tot_prod = []
    for i in range(num_run_iterations):
        # Take a step as suggested by the agent
        act = agent.act(obs, reward=0)
        obs, reward, done, _ = environment.step(act)

        # Collect statistics
        cum_reward += reward
        tot_load.append(sum(obs.load_p))
        tot_prod.append(sum(obs.prod_p))
        print('i:', i, '\tact:', 0, '\treward:', reward, '\ttot load', tot_load[i], '\ttot prod', tot_prod[i])
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


def run_sac_new(env, agent, num_run_iterations):
    """ Run the SAC agent and print/plot/write statistics."""
    # Create Tensorboard writer
    rundir = os.path.join('runs', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    run_writer = tf.summary.create_file_writer(rundir, name=agent.name)

    # Run the agent
    obs = env.reset()
    cum_reward = 0.
    act_old = None
    proposed_acts = []
    tot_load = []
    tot_prod = []
    for i in range(num_run_iterations):
        # Get suggested acts
        acts, probs = agent.my_acts(agent.convert_obs(obs), nr_acts=1)

        act = acts[0]
        # Simulate the top proposed actions and pick the top one.
        # rw_to_beat = -100000
        # for act_proposal in acts:
        #    _, rw_proposal, _, _ = obs.simulate(agent.convert_act(act_proposal))
        #    if rw_proposal > rw_to_beat:
        #        act = act_proposal
        #        rw_to_beat = rw_proposal

        if act != act_old:
            print(agent.convert_act(act))
            act_old = act

        # Take a step
        obs, reward, done, _ = env.step(agent.convert_act(act))

        # Collect statistics
        proposed_acts.append(acts)
        cum_reward += reward
        tot_load.append(sum(obs.load_p))
        tot_prod.append(sum(obs.prod_p))
        print('i:', i, '\tact:', act, '\treward:', reward, '\tproposed acts:', acts, tot_load[i], '\ttot prod', tot_prod[i])
        with run_writer.as_default():
            tf.summary.scalar("run/reward", reward, i)
            tf.summary.scalar("run/act", act, i)
            tf.summary.scalar("run/cum_reward", cum_reward, i)

        if done:
            env.reset()
            print('Total reward: ', i, cum_reward)
            break

    # Plot the total production and load encountered in the run
    plt.plot(tot_prod)
    plt.plot(tot_load)
    plt.legend(['Total production', 'Total load'])
    plt.xlabel('timestep')
    plt.ylabel('Power [MW]')
    plt.show()

    # Plot histogram of the proposed acts to see how they are distributed.
    proposed_acts_np = np.array(proposed_acts).flatten()
    plt.hist(proposed_acts_np)
    plt.show()

    # Print the unique actions
    for act in np.unique(proposed_acts_np):
        print(act, agent.convert_act(act))


def plot_grid_layout(environment, save_file_path=None):
    """Plot the grid layout."""
    plot_helper = PlotMatplot(environment.observation_space)
    fig_layout = plot_helper.plot_layout()
    plt.show(fig_layout)
    if save_file_path is not None:
        plt.savefig(fname=save_file_path)


def main():
    # SETTINGS ========================================
    NUM_TRAIN_ITERATIONS = 5000
    NUM_RUN_ITERATIONS = 5000
    path_grid = 'rte_case14_redisp'
    save_path = "saved_networks"

    train_agent = False
    run_agent = False  # Make sure the
    run_agent_path = r'C:\Users\johan\Documents\GitHub\AIDML2RPN\src\saved_networks\rte_case14_redisp_SACAgentNew_5000'
    run_name = 'SacAgentNew'
    # ==================================================

    if train_agent:
        # Initialize the environment and agent
        environment = make(path_grid, reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)
        agent = SACAgent(action_space=environment.action_space, name='SACAgentNew')

        # Paths for the logs and for saving the network
        logdir = os.path.join('logs2', agent.name, datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
        network_path = os.path.join(save_path, '{}_{}_{}'.format(path_grid, agent.name, NUM_TRAIN_ITERATIONS))
        for path in [logdir, network_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Train the agent for NUM_TRAIN_ITERATION steps
        agent.train(environment, NUM_TRAIN_ITERATIONS, network_path, logdir=logdir, training_param=TrainingParamSAC())

    if run_agent:
        env = make('rte_case14_realistic', reward_class=L2RPNReward, action_class=TopologyChangeAction, test=False)

        # Load a SAC agent from the path run_agent_path
        agent = SACAgent(action_space=env.action_space, name=run_name)
        obs = env.reset()
        transformed_obs = agent.convert_obs(obs)
        agent.init_deep_q(transformed_obs)
        agent.deep_q.load_network(run_agent_path)

        # Run the SAC agent
        run_sac_new(env, agent, NUM_RUN_ITERATIONS)

        # Run do-nothing agent
        run_do_nothing(env, NUM_RUN_ITERATIONS)

        # Plot grid
        plot_grid_layout(env)


if __name__ == "__main__":
    main()
