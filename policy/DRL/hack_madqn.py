import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import copy

print mx.gpu()
a = nd.array([1, 2, 3], ctx=mx.gpu())

CTX = mx.gpu()


class MultiAgentNetwork(gl.nn.Block):
    def __init__(self, domain_string, hidden_layers, local_hidden_units, global_hidden_units, **kwargs):
        gl.nn.Block.__init__(self, **kwargs)

        self.domain_string = domain_string
        if self.domain_string == 'Laptops11':
            self.slots = ['batteryrating', 'driverange', 'family',
                          'isforbusinesscomputing', 'platform', 'pricerange',
                          'processorclass', 'sysmemory', 'utility', 'warranty',
                          'weightrange']  # output order
            self.slot_dimension = {
                'warranty': (0, 5),
                'family': (130, 136),
                'utility': (136, 145),
                'platform': (145, 152),
                'processorclass': (152, 164),
                'pricerange': (164, 169),
                'batteryrating': (169, 174),
                'sysmemory': (174, 183),
                'weightrange': (183, 188),
                'isforbusinesscomputing': (188, 192),
                'driverange': (192, 197)
            }
            self.global_dimension = [(5, 130), (197, 257)]
            self.input_dimension = 257
            self.global_input_dimension = 257 - 197 + 125
        else:
            raise ValueError

        self.hidden_layers = hidden_layers
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units

        with self.name_scope():
            self.input_trans = {}
            for slot in self.slots:
                in_units = self.slot_dimension[slot][1] - self.slot_dimension[slot][0]
                self.input_trans[slot] = gl.nn.Dense(in_units=in_units, units=self.local_hidden_units[0], activation='relu')
                self.register_child(self.input_trans[slot])
            self.input_trans['global'] = \
                gl.nn.Dense(in_units=self.global_input_dimension, units=self.global_hidden_units[0], activation='relu')
            self.register_child(self.input_trans['global'])

            self.local_share_trans = []
            self.global_trans = []
            self.local2local_comm = []
            self.local2global_comm = []
            self.global2local_comm = []
            for i in range(self.hidden_layers - 1):
                self.local_share_trans.append(gl.nn.Dense(in_units=self.local_hidden_units[i],
                                                          units=self.local_hidden_units[i + 1], activation='relu'))
                self.global_trans.append(gl.nn.Dense(in_units=self.global_hidden_units[i],
                                                     units=self.global_hidden_units[i + 1], activation='relu'))
                self.local2local_comm.append(gl.nn.Dense(in_units=self.local_hidden_units[i],
                                                         units=self.local_hidden_units[i + 1], activation='relu'))
                self.local2global_comm.append(gl.nn.Dense(in_units=self.local_hidden_units[i],
                                                          units=self.global_hidden_units[i + 1], activation='relu'))
                self.global2local_comm.append(gl.nn.Dense(in_units=self.global_hidden_units[i],
                                                          units=self.local_hidden_units[i + 1], activation='relu'))
                self.register_child(self.local_share_trans[-1])
                self.register_child(self.global_trans[-1])
                self.register_child(self.local2local_comm[-1])
                self.register_child(self.local2global_comm[-1])
                self.register_child(self.global2local_comm[-1])

            self.output_trans = []
            for i in range(len(self.slots)):
                self.output_trans.append(gl.nn.Dense(in_units=self.local_hidden_units[-1], units=3))
                self.register_child(self.output_trans[-1])
            self.output_trans.append(gl.nn.Dense(in_units=self.global_hidden_units[-1], units=7))
            self.register_child(self.output_trans[-1])

    def forward(self, input_vec):
        inputs = {}
        for slot in self.slots:
            inputs[slot] = input_vec[:, self.slot_dimension[slot][0]:self.slot_dimension[slot][1]]
        input_global = []
        for seg in self.global_dimension:
            input_global.append(input_vec[:, seg[0]:seg[1]])
        inputs['global'] = nd.concat(*input_global, dim=1)

        layer = [[], ]
        for slot in self.slots:
            layer[0].append(self.input_trans[slot](inputs[slot]))
        layer[0].append(self.input_trans['global'](inputs['global']))

        for i in range(self.hidden_layers - 1):
            layer.append([])
            for j in range(len(self.slots)):
                layer[i + 1].append(self.local_share_trans[i](layer[i][j]))
            layer[i + 1].append(self.global_trans[i](layer[i][-1]))

            for j in range(len(self.slots)):
                for k in range(len(self.slots)):
                    if j != k:
                        layer[i + 1][k] = layer[i + 1][k] + self.local2local_comm[i](layer[i][j])
                layer[i + 1][-1] = layer[i + 1][-1] + self.local2global_comm[i](layer[i][j])
            for j in range(len(self.slots)):
                layer[i + 1][j] = layer[i + 1][j] + self.global2local_comm[i](layer[i][-1])

        outputs = []
        for i in range(len(self.slots) + 1):
            outputs.append(self.output_trans[i](layer[-1][i]))
        return nd.concat(*outputs, dim=1)


# ===========================
#   Deep Q Network
# ===========================
class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """

    def __init__(self, state_dim, action_dim, learning_rate, tau,
                 num_actor_vars, minibatch_size=64, architecture='duel',
                 h1_size=130, h1_drop=None, h2_size=50, h2_drop=None, domain_string=None):
        # self.sess = sess
        self.domain_string = domain_string
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
        network =  MultiAgentNetwork(domain_string=self.domain_string, hidden_layers=2,
                                     local_hidden_units=(40, 20),
                                     global_hidden_units=(self.h1_size, self.h2_size),
                                     prefix=prefix)
        network.initialize(ctx=CTX)
        return network

    def train(self, inputs, action, sampled_q):
        inputs = copy.deepcopy(inputs)
        action = copy.deepcopy(action)
        sampled_q = copy.deepcopy(sampled_q)

        inputs = nd.array(inputs, ctx=CTX)
        action = nd.array(action, ctx=CTX)
        sampled_q = nd.array(sampled_q, ctx=CTX)
        sampled_q = sampled_q.reshape(shape=(sampled_q.shape[0],))

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
        return self.qnet(nd.array(inputs, ctx=CTX)).asnumpy()

    def predict_target(self, inputs):
        return self.target(nd.array(inputs, ctx=CTX)).asnumpy()

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
            self.qnet.load_params(filename=load_filename + '_qnet',
                                  ctx=CTX)
            self.target.load_params(filename=load_filename + '_target',
                                    ctx=CTX)
            print "Successfully loaded:", load_filename
        except:
            print "Could not find old network weights"
            print load_filename

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.qnet.save_params(filename=save_filename + '_qnet')
        self.target.save_params(filename=save_filename + '_target')
