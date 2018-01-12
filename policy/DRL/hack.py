"""
Implementation of DQN -  Deep Q Network

The algorithm is developed with tflearn + Tensorflow

Author: Pei-Hao Su
"""
import tensorflow as tf
import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import copy

# ===========================
#   Deep Q Network
# ===========================
class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, state_dim, action_dim, learning_rate, tau,
                 num_actor_vars, minibatch_size=64, architecture='duel',
                 h1_size=130, h1_drop=None, h2_size=50, h2_drop=None):
        # self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h1_drop = h1_drop
        self.h2_size = h2_size
        self.h2_drop = h2_drop
        self.minibatch_size = minibatch_size

        self.qnet = self.create_ddq_network(prefix='qnet_')
        self.target = self.create_ddq_network(prefix='target_')

        self.trainer = gl.Trainer(params=self.qnet.collect_params(),
                                  optimizer='adam',
                                  optimizer_params=dict(learning_rate=self.learning_rate))

    def create_ddq_network(self, prefix=''):
        network = gl.nn.Sequential(prefix=prefix)
        with network.name_scope():
            network.add(
                gl.nn.Dense(in_units=self.s_dim, units=self.h1_size, activation='relu'),
                gl.nn.Dropout(rate=self.h1_drop),
                gl.nn.Dense(in_units=self.h1_size, units=self.h2_size, activation='relu'),
                gl.nn.Dropout(rate=self.h2_drop),
                gl.nn.Dense(in_units=self.h2_size, units=self.a_dim)
            )
        network.initialize()
        return network

    def train(self, inputs, action, sampled_q):
        inputs = copy.deepcopy(inputs)
        action = copy.deepcopy(action)
        sampled_q = copy.deepcopy(sampled_q)

        inputs = nd.array(inputs)
        action = nd.array(action)
        sampled_q = nd.array(sampled_q)
        sampled_q = sampled_q.reshape(shape=(sampled_q.shape[0], ))

        with mx.autograd.record():
            outputs = self.qnet(inputs)
            td_error = nd.sum(data=outputs * action, axis=1) - sampled_q
            loss = 0
            for i in range(self.minibatch_size):
                if nd.abs(td_error[i]) < 1.0:
                    loss = loss + 0.5 * nd.square(td_error[i])
                else:
                    loss = loss + nd.abs(td_error[i]) - 0.5
        loss.backward()
        self.trainer.step(batch_size=self.minibatch_size)

    def predict(self, inputs):
        return self.qnet(nd.array(inputs)).asnumpy()

    def predict_target(self, inputs):
        return self.target(nd.array(inputs)).asnumpy()

    def update_target_network(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            assert (param_list_target[i].name.strip('target') ==
                    param_list_qnet[i].name.strip('qnet'))
            param_list_target[i].set_data(
                param_list_target[i].data() * (1. - self.tau) +
                param_list_qnet[i].data() * self.tau
            )

    def copy_qnet_to_target(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            assert (param_list_target[i].name.strip('target') ==
                    param_list_qnet[i].name.strip('qnet'))
            param_list_target[i].set_data(param_list_qnet[i].data())

    def load_network(self, load_filename):
        try:
            self.qnet.load_params(filename=load_filename + '_qnet', ctx=mx.cpu())
            self.target.load_params(filename=load_filename + '_target', ctx=mx.cpu())
            print "Successfully loaded:", load_filename
        except:
            print "Could not find old network weights"
            print load_filename

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.qnet.save_params(filename=save_filename + '_qnet')
        self.target.save_params(filename=save_filename + '_target')