import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.utils import TrainingParam


class SACNetwork(SAC_NN):

    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 learning_rate_decay_steps=1000,
                 learning_rate_decay_rate=0.95,
                 training_param=TrainingParam()):
        super().__init__(action_size,
                         observation_size,
                         lr,
                         learning_rate_decay_steps,
                         learning_rate_decay_rate,
                         training_param)

    def predict_movement(self, data, epsilon, batch_size=None):
        if batch_size is None:
            batch_size = data.shape[0]
        rand_val = np.random.random(data.shape[0])
        p_actions = self.model_policy.predict(data, batch_size=batch_size)

        # TODO: change to stochastic policy
        opt_policy_orig = np.argmax(np.abs(p_actions), axis=-1)
        opt_policy = 1.0 * opt_policy_orig
        opt_policy[rand_val < epsilon] = np.random.randint(0, self.action_size, size=(np.sum(rand_val < epsilon)))

        # store the qvalue_evolution (lots of computation time maybe here)
        # opt_policy_orig_ts = tf.convert_to_tensor(opt_policy_orig, dtype=tf.int32)
        # tmp, previous_arange = self.get_eye_pm(data.shape[0])
        # tmp[previous_arange, opt_policy_orig] = 1.0
        # tmp_ts = tf.convert_to_tensor(tmp, dtype=tf.float32)
        # q_actions0 = self.model_Q((data, tmp_ts)).numpy()
        # q_actions2 = self.model_Q2((data, tmp_ts)).numpy()
        # tmp[previous_arange, opt_policy_orig] = 0.0
        #
        # q_actions = np.fmin(q_actions0, q_actions2).reshape(-1)
        # self.qvalue_evolution = np.concatenate((self.qvalue_evolution, q_actions))
        # above is not mandatory for predicting a movement so, might need to be moved somewhere else...
        opt_policy = opt_policy.astype(np.int)
        return opt_policy, p_actions[:, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """Trains networks to fit given parameters"""
        if batch_size is None:
            batch_size = s_batch.shape[0]
        target = np.zeros((batch_size, 1))
        # training of the action state value networks
        last_action = np.zeros((batch_size, self.action_size))
        # Save the graph just the first time
        if tf_writer is not None:
            tf.summary.trace_on()
        fut_action = self.model_value_target.predict(s2_batch, batch_size=batch_size).reshape(-1)
        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.trace_export("model_value_target-graph", 0)
            tf.summary.trace_off()

        target[:, 0] = r_batch + (1 - d_batch) * self.training_param.DECAY_RATE * fut_action
        loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        loss_2 = self.model_Q2.train_on_batch([s_batch, last_action], target)

        self.life_spent += 1
        temp = 1 / np.log(self.life_spent) / 2
        tiled_batch = np.tile(s_batch, (self.action_size, 1))
        tiled_batch_ts = tf.convert_to_tensor(tiled_batch)
        # tiled_batch: output something like: batch, batch, batch
        # TODO save that somewhere not to compute it each time, you can even save this in the
        # TODO tensorflow graph!
        tmp = self.get_eye_train(batch_size)
        # tmp is something like [1,0,0] (batch size times), [0,1,0,...] batch size time etc.

        action_v1_orig = self.model_Q.predict([tiled_batch_ts, tmp], batch_size=batch_size).reshape(batch_size, -1)
        action_v2_orig = self.model_Q2.predict([tiled_batch_ts, tmp], batch_size=batch_size).reshape(batch_size, -1)
        action_v1 = action_v1_orig - np.amax(action_v1_orig, axis=-1).reshape(batch_size, 1)
        new_proba = np.exp(action_v1 / temp) / np.sum(np.exp(action_v1 / temp), axis=-1).reshape(batch_size, 1)
        new_proba_ts = tf.convert_to_tensor(new_proba)
        loss_policy = self.model_policy.train_on_batch(s_batch, new_proba_ts)

        # training of the value_function
        # if tf_writer is not None:
        #     tf.summary.trace_on()
        target_pi = self.model_policy.predict(s_batch, batch_size=batch_size)
        # if tf_writer is not None:
        #     with tf_writer.as_default():
        #         tf.summary.trace_export("model_policy-graph", 0)
        #     tf.summary.trace_off()
        value_target = np.fmin(action_v1_orig[0, a_batch], action_v2_orig[0, a_batch]) - np.sum(
            target_pi * np.log(target_pi + 1e-6))
        value_target_ts = tf.convert_to_tensor(value_target.reshape(-1, 1))
        loss_value = self.model_value.train_on_batch(s_batch, value_target_ts)

        self.Is_nan = np.isnan(loss) + np.isnan(loss_2) + np.isnan(loss_policy) + np.isnan(loss_value)
        return np.all(np.isfinite(loss)) & np.all(np.isfinite(loss_2)) & np.all(np.isfinite(loss_policy)) & \
               np.all(np.isfinite(loss_value))

    def summary(self):
        stringlist = []
        stringlist.append('model_value')
        self.model_value.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_Q')
        self.model_Q.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append('model_policy')
        self.model_policy.summary(print_fn=lambda x: stringlist.append(x))

        short_model_summary = "\n".join(stringlist)
        return short_model_summary

