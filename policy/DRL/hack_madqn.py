import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import copy
import random

print mx.gpu()
a = nd.array([1, 2, 3], ctx=mx.gpu())
# b = nd.array([7, 8, 9], ctx=mx.gpu())
# print a + b

CTX = mx.gpu()


class MATransfer(gl.nn.Block):
    def __init__(self, slots, local_in_units, local_units, local_dropout,
                 global_in_units, global_units, global_dropout, activation,
                 concrete_share_rate, dropout_regularizer, use_comm=True):
        gl.nn.Block.__init__(self)
        self.slots = slots
        self.use_comm = use_comm
        self.local_in_units = local_in_units
        self.local_units = local_units
        self.concrete_share_rate = concrete_share_rate
        self.dropout_regularizer = dropout_regularizer
        with self.name_scope():
            self.local_share_trans = gl.nn.Dense(in_units=local_in_units, units=local_units, activation=activation)
            self.global_trans = gl.nn.Dense(in_units=global_in_units, units=global_units, activation=activation)
            if self.use_comm:
                self.local2local_share_comm = gl.nn.Dense(
                    in_units=local_in_units, units=local_units, activation=activation)
                self.local2global_comm = gl.nn.Dense(in_units=local_in_units, units=global_units, activation=activation)
                self.global2local_comm = gl.nn.Dense(in_units=global_in_units, units=local_units, activation=activation)
            self.local_private_trans = []
            self.local2local_private_comm = []
            self.share_rate = []
            for i in range(self.slots):
                self.local_private_trans.append(gl.nn.Dense(
                    in_units=local_in_units, units=local_units, activation=activation))
                self.register_child(self.local_private_trans[-1])
                if self.use_comm:
                    self.local2local_private_comm.append(gl.nn.Dense(
                        in_units=local_in_units, units=local_units, activation=activation))
                    self.register_child(self.local2local_private_comm[-1])
                if concrete_share_rate:
                    self.share_rate.append(self.params.get(
                        name='sharerate_{slot_num}'.format(slot_num=i), shape=(1, 1), init=mx.init.Constant([[2.]])))

            self.local_dropout_op = gl.nn.Dropout(local_dropout)
            self.global_dropout_op = gl.nn.Dropout(global_dropout)

        # print concrete_share_rate
        # print dropout_regularizer
        # exit(0)

    def forward(self, inputs, loss=None):
        assert len(inputs) == self.slots + 1

        local_drop_vec = nd.ones_like(inputs[0])
        local_drop_vec = self.local_dropout_op(local_drop_vec)
        for i in range(self.slots):
            inputs[i] = inputs[i] * local_drop_vec
        inputs[-1] = self.global_dropout_op(inputs[-1])

        local_share_vec = []
        local_private_vec = []
        if self.concrete_share_rate:
            for i in range(self.slots):
                proba = nd.sigmoid(data=self.share_rate[i].data())
                proba = nd.broadcast_axis(data=proba, axis=(0, 1), size=inputs[0].shape)
                u_vec = nd.random_uniform(low=1e-5, high=1. - 1e-5, shape=inputs[0].shape, ctx=CTX)
                local_share_vec.append(nd.sigmoid(10. * (
                    nd.log(proba) - nd.log(1. - proba) +
                    nd.log(u_vec) - nd.log(1. - u_vec)
                )))
                local_private_vec.append(1. - local_share_vec[i])
                # print 'proba:', proba
                # print 'dropout_regularizer:', self.dropout_regularizer
                if loss is not None:
                    loss.append(
                        self.dropout_regularizer * nd.sum(proba * nd.log(proba) + (1. - proba) * nd.log(1. - proba)))
            if random.random() < 0.01:
                for i in range(self.slots):
                    proba = nd.sigmoid(data=self.share_rate[i].data())
                    print proba.asnumpy(),
                print ''

        else:
            local_share_vec = [nd.ones_like(inputs[0]), ] * self.slots
            local_private_vec = [nd.zeros_like(inputs[0]), ] * self.slots
        # local_share_vec = (1. - self.private_rate) * nd.Dropout(
        #     nd.ones(shape=(inputs[0].shape[0], self.local_units)), p=self.private_rate, mode='always')
        # local_private_vec = 1. - local_share_vec

        results = []

        for i in range(self.slots):
            results.append(self.local_private_trans[i](local_share_vec[i] * inputs[i]) +
                           self.local_share_trans(local_private_vec[i] * inputs[i]))
        results.append(self.global_trans(inputs[-1]))

        if self.use_comm:
            mean_vec = nd.zeros_like(inputs[0])
            for i in range(self.slots):
                mean_vec = mean_vec + inputs[i]
            mean_vec = mean_vec / float(self.slots)

            for i in range(self.slots):
                results[i] = results[i] + self.local2local_share_comm(local_share_vec[i] * mean_vec) + \
                             self.local2local_private_comm[i](local_private_vec[i] * mean_vec)
                results[i] = results[i] + self.global2local_comm(inputs[-1])
            results[-1] = results[-1] + self.local2global_comm(mean_vec)
            # for i in range(self.slots):
            #     for j in range(self.slots):
            #         if i != j:
            #             results[j] = results[j] + self.local2local_share_comm(inputs[i])
            #     results[-1] = results[-1] + self.local2global_comm(inputs[i])
            # for i in range(self.slots):
            #     results[i] = results[i] + self.global2local_comm(inputs[-1])

        return results


class MultiAgentNetwork(gl.nn.Block):
    def __init__(self, domain_string, hidden_layers, local_hidden_units, local_dropouts,
                 global_hidden_units, global_dropouts, private_rate, sort_input_vec,
                 share_last_layer, recurrent_mode, input_comm, concrete_share_rate, dropout_regularizer, **kwargs):
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
        self.local_dropouts = local_dropouts
        self.global_hidden_units = global_hidden_units
        self.global_dropouts = global_dropouts
        self.private_rate = private_rate
        self.sort_input_vec = sort_input_vec
        self.share_last_layer = share_last_layer
        self.recurrent_mode = recurrent_mode
        self.conrete_share_rate = concrete_share_rate
        self.dropout_regularizer = dropout_regularizer

        with self.name_scope():
            if self.sort_input_vec is False:
                self.input_trans = {}
                for slot in self.slots:
                    in_units = self.slot_dimension[slot][1] - self.slot_dimension[slot][0]
                    self.input_trans[slot] = \
                        gl.nn.Dense(in_units=in_units, units=self.local_hidden_units[0], activation='relu')
                    self.register_child(self.input_trans[slot])
                self.input_trans['global'] = \
                    gl.nn.Dense(in_units=self.global_input_dimension,
                                units=self.global_hidden_units[0], activation='relu')
                self.register_child(self.input_trans['global'])
            else:
                self.input_trans = MATransfer(
                    slots=len(self.slots),
                    local_in_units=22,
                    local_units=self.local_hidden_units[0],
                    local_dropout=0.,
                    global_in_units=self.global_input_dimension,
                    global_units=self.global_hidden_units[0],
                    global_dropout=0.,
                    activation='relu',
                    use_comm=input_comm,
                    concrete_share_rate=self.conrete_share_rate,
                    dropout_regularizer=self.dropout_regularizer
                )

            if self.recurrent_mode is False:
                self.ma_trans = []
                for i in range(self.hidden_layers - 1):
                    self.ma_trans.append(MATransfer(
                        slots=len(self.slots),
                        local_in_units=self.local_hidden_units[i],
                        local_units=self.local_hidden_units[i + 1],
                        local_dropout=self.local_dropouts[i],
                        global_in_units=self.global_hidden_units[i],
                        global_units=self.global_hidden_units[i + 1],
                        global_dropout=self.global_dropouts[i],
                        activation='relu',
                        concrete_share_rate=self.conrete_share_rate,
                        dropout_regularizer=self.dropout_regularizer
                    ))
                    self.register_child(self.ma_trans[-1])
            else:
                assert self.local_hidden_units == (self.local_hidden_units[0], ) * self.hidden_layers
                assert self.local_dropouts == (self.local_dropouts[0], ) * self.hidden_layers
                assert self.global_hidden_units == (self.global_hidden_units[0], ) * self.hidden_layers
                assert self.global_dropouts == (self.global_dropouts[0], ) * self.hidden_layers

                self.ma_trans = MATransfer(
                        slots=len(self.slots),
                        local_in_units=self.local_hidden_units[0],
                        local_units=self.local_hidden_units[0],
                        local_dropout=self.local_dropouts[0],
                        global_in_units=self.global_hidden_units[0],
                        global_units=self.global_hidden_units[0],
                        global_dropout=self.global_dropouts[0],
                        activation='relu',
                        concrete_share_rate=self.conrete_share_rate,
                        dropout_regularizer=self.dropout_regularizer
                )

            if self.share_last_layer is False:
                self.local_out_drop_op = gl.nn.Dropout(self.local_dropouts[-1])
                self.global_out_drop_op = gl.nn.Dropout(self.global_dropouts[-1])

                self.output_trans = []
                for i in range(len(self.slots)):
                    self.output_trans.append(gl.nn.Dense(in_units=self.local_hidden_units[-1], units=3))
                    self.register_child(self.output_trans[-1])
                self.output_trans.append(gl.nn.Dense(in_units=self.global_hidden_units[-1], units=7))
                self.register_child(self.output_trans[-1])
            else:
                self.output_trans = MATransfer(
                    slots=len(self.slots),
                    local_in_units=self.local_hidden_units[-1],
                    local_units=3,
                    local_dropout=self.local_dropouts[self.hidden_layers - 1],
                    global_in_units=self.global_hidden_units[-1],
                    global_units=7,
                    global_dropout=self.global_dropouts[self.hidden_layers - 1],
                    activation=None,
                    concrete_share_rate=self.conrete_share_rate,
                    dropout_regularizer=self.dropout_regularizer
                )

            # for key, value in self.collect_params().items():
            #     print key, value
            # exit(0)

    def forward(self, input_vec, loss=None):
        # get inputs for every slot(including global)
        inputs = {}
        for slot in self.slots:
            inputs[slot] = input_vec[:, self.slot_dimension[slot][0]:self.slot_dimension[slot][1]]
        input_global = []
        for seg in self.global_dimension:
            input_global.append(input_vec[:, seg[0]:seg[1]])
        inputs['global'] = nd.concat(*input_global, dim=1)

        layer = []
        # inputs -> first_hidden_layer
        if self.sort_input_vec is False:
            layer.append([])
            for slot in self.slots:
                layer[0].append(self.input_trans[slot](inputs[slot]))
            layer[0].append(self.input_trans['global'](inputs['global']))
        else:
            sorted_inputs = []
            for slot in self.slots:
                tmp = inputs[slot][:, :-2].sort(is_ascend=False)
                if tmp.shape[1] < 20:
                    tmp = nd.concat(tmp, nd.zeros((tmp.shape[0], 20 - tmp.shape[1]), ctx=CTX), dim=1)
                else:
                    raise ValueError
                sorted_inputs.append(nd.concat(tmp, inputs[slot][:, -2:], dim=1))
            sorted_inputs.append(inputs['global'])
            layer.append(self.input_trans(sorted_inputs, loss))

        # hidden_layers
        for i in range(self.hidden_layers - 1):
            if self.recurrent_mode is False:
                layer.append(self.ma_trans[i](layer[i], loss))
            else:
                layer.append(self.ma_trans(layer[i], loss))

        if self.share_last_layer is False:
            # dropout of last hidden layer
            for j in range(len(self.slots)):
                layer[-1][j] = self.local_out_drop_op(layer[-1][j])
            layer[-1][-1] = self.global_out_drop_op(layer[-1][-1])

            # last_hidden_layer -> outputs
            outputs = []
            for i in range(len(self.slots) + 1):
                outputs.append(self.output_trans[i](layer[-1][i]))
        else:
            outputs = self.output_trans(layer[-1], loss)
        return nd.concat(*outputs, dim=1)


class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """

    def __init__(self, state_dim, action_dim, learning_rate, tau,
                 num_actor_vars, minibatch_size=64, architecture='duel',
                 h1_size=130, h1_drop=None, h2_size=50, h2_drop=None, domain_string=None,
                 hidden_layers=None, local_hidden_units=None, local_dropouts=None,
                 global_hidden_units=None, global_dropouts=None, private_rate=None,
                 sort_input_vec=None, share_last_layer=None, recurrent_mode=None,
                 input_comm=None, target_explore=None, concrete_share_rate=None,
                 dropout_regularizer=None, weight_regularizer=None):
        self.domain_string = domain_string
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.hidden_layers = hidden_layers
        self.local_hidden_units = local_hidden_units
        self.local_dropouts = local_dropouts
        self.global_hidden_units = global_hidden_units
        self.global_dropouts = global_dropouts
        self.minibatch_size = minibatch_size
        self.private_rate = private_rate
        self.sort_input_vec = sort_input_vec
        self.share_last_layer = share_last_layer
        self.recurrent_mode = recurrent_mode
        self.input_comm = input_comm
        self.target_explore = target_explore
        self.concrete_share_rate = concrete_share_rate
        self.dropout_regularizer = dropout_regularizer

        self.qnet = self.create_ddq_network(prefix='qnet_')
        self.target = self.create_ddq_network(prefix='target_')

        self.trainer = gl.Trainer(params=self.qnet.collect_params(), optimizer='adam',
                                  optimizer_params=dict(learning_rate=self.learning_rate, wd=weight_regularizer))

    def create_ddq_network(self, prefix=''):
        network = MultiAgentNetwork(domain_string=self.domain_string,
                                    hidden_layers=self.hidden_layers,
                                    local_hidden_units=self.local_hidden_units,
                                    local_dropouts=self.local_dropouts,
                                    global_hidden_units=self.global_hidden_units,
                                    global_dropouts=self.local_dropouts,
                                    private_rate=self.private_rate,
                                    sort_input_vec=self.sort_input_vec,
                                    share_last_layer=self.share_last_layer,
                                    recurrent_mode=self.recurrent_mode,
                                    input_comm=self.input_comm,
                                    concrete_share_rate=self.concrete_share_rate,
                                    dropout_regularizer=self.dropout_regularizer,
                                    prefix=prefix)
        # print network.collect_params()
        # exit(0)
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
            loss_vec = []
            outputs = self.qnet(inputs, loss_vec)
            loss = 0.
            for element in loss_vec:
                loss = loss + element
            print 'loss_dropout:', loss
            td_error = nd.sum(data=outputs * action, axis=1) - sampled_q
            for i in range(self.minibatch_size):
                if nd.abs(td_error[i]) < 1.0:
                    loss = loss + 0.5 * nd.square(td_error[i])
                else:
                    loss = loss + nd.abs(td_error[i]) - 0.5
            print loss
        loss.backward()
        self.trainer.step(batch_size=self.minibatch_size)

    def predict(self, inputs):
        return self.qnet(nd.array(inputs, ctx=CTX)).asnumpy()

    def predict_target(self, inputs):
        if self.target_explore:
            with mx.autograd.train_mode():
                return self.target(nd.array(inputs, ctx=CTX)).asnumpy()
        else:
            assert mx.autograd.is_training() == False
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
            assert (param_list_target[i].name.strip('target') == param_list_qnet[i].name.strip('qnet'))
            param_list_target[i].set_data(param_list_target[i].data() * (1. - self.tau) +
                                          param_list_qnet[i].data() * self.tau)

    def copy_qnet_to_target(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            assert param_list_target[i].name.strip('target') == param_list_qnet[i].name.strip('qnet')
            param_list_target[i].set_data(param_list_qnet[i].data())

    def load_network(self, load_filename):
        try:
            self.qnet.load_params(filename=load_filename + '_qnet', ctx=CTX)
            print "Successfully loaded:", load_filename + '_qnet'
        except:
            print "Could not find old network weights(qnet)"
            print load_filename

        try:
            self.target.load_params(filename=load_filename + '_target', ctx=CTX)
            print "Successfully loaded:", load_filename + '_target'
        except:
            print "Could not find old network weights(target)"
            print load_filename

        try:
            self.trainer.step(1, ignore_stale_grad=True)
            self.trainer.load_states(fname=load_filename + '_trainer')
            print "Successfully loaded:", load_filename + '_trainer'
        except:
            print "Could not find old network weights(trainer)"
            print load_filename

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.qnet.save_params(filename=save_filename + '_qnet')
        self.target.save_params(filename=save_filename + '_target')
        self.trainer.save_states(fname=save_filename + '_trainer')
