#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import os
import paddle as pd
device = pd.set_device('gpu')
import time
import unittest
pd.disable_static()


#from mixprec import float32_variable_storage_getter, LossScalingOptimizer

'''
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
         if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)
'''

# Class holding statistics
class Stats:
    def __init__(self):
        self.s = {}
    def add(self, stat_dict):
        for (k,v) in stat_dict.items():
            if k not in self.s:
                self.s[k] = []
            self.s[k].append(v)
    def n(self, name):
        return len(self.s[name] or [])
    def mean(self, name):
        return np.mean(self.s[name] or [0])
    def stddev_mean(self, name):
        # standard deviation in the sample mean.
        return math.sqrt(
            np.var(self.s[name] or [0]) / max(0.0001, (len(self.s[name]) - 1)))
    def str(self):
        return ', '.join(
            ["{}={:g}".format(k, np.mean(v or [0])) for k,v in self.s.items()])
    def clear(self):
        self.s = {}
    def summaries(self, tags):
        return [tf.Summary.Value(
            tag=k, simple_value=self.mean(v)) for k,v in tags.items()]

# Simple timer
class Timer:
    def __init__(self):
        self.last = time.time()
    def elapsed(self):
        # Return time since last call to 'elapsed()'
        t = time.time()
        e = t - self.last
        self.last = t
        return e

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 首先实现中间两个卷积层，Skip Connection 1x1 卷积层的残差模块。代码如下：
# 残差模块

class Conv_Block(nn.Layer):
    def __init__(self, in_channel, out_channel,  kernel_size=3, stride=1, padding='SAME'):
        super(Conv_Block, self).__init__()
        
        # 第一个卷积单元
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        # 前向计算
        # [b, c, h, w], 通过第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Res_Block(nn.Layer):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Res_Block, self).__init__()
        
        # 第一个卷积单元
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.relu = nn.ReLU()

        # 第二个卷积单元
        self.conv2 = nn.Conv2D(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(out_channel)

    def forward(self, x):
        # 前向计算
        # [b, c, h, w], 通过第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        #  2 条路径输出直接相加,然后输入激活函数
        output = F.relu(out + x)

        return output

def build_trunk(in_channel, out_channel, num_layers):

    block_list = []
    for i in range(num_layers):

        block_list.append(Res_Block(out_channel, out_channel))

    trunk = nn.Sequential(*block_list) #用*号可以把list列表展开为元素
    return trunk

class ResNet(nn.Layer):
    # 继承paddle.nn.Layer定义网络结构
    def __init__(self, residual_blocks=20, residual_filters=256):
        super(ResNet, self).__init__()
        # 初始化函数(根网络，预处理)
        # x:[b, c, h ,w]=[b,18,19,19]
        self.flow = Conv_Block(in_channel=18, out_channel=residual_filters, kernel_size=3, 
            stride=1, padding='SAME')# 第一层卷积,x:[b,18,256,256]
  
        self.trunk = build_trunk(residual_filters, residual_filters, residual_blocks) # x:[b,256,19,19]
        # policy head

        self.conv_pol =  Conv_Block(in_channel=residual_filters, out_channel=2, kernel_size=1, 
            stride=1, padding='SAME')

        # value head 
        self.conv_val = Conv_Block(in_channel=residual_filters, out_channel=1, kernel_size=1, 
            stride=1, padding='SAME')


        self.fc_pol = nn.Linear(in_features=2*19*19,out_features=19*19+1)
        self.fc_val_1 = nn.Linear(in_features=19*19,out_features=256)
        self.fc_val_2 = nn.Linear(in_features=256,out_features=1)
  
        self.flatten = nn.Flatten()


    def forward(self, inputs):
        # 前向计算函数：通过根网络
        x = self.flow(inputs)
        x = self.trunk(x)

        # policy head
        pol = self.conv_pol(x)
        pol = self.flatten(pol)
        policy_output = self.fc_pol(pol)

        # value head 
        val = self.conv_val(x)
        val = self.flatten(val)
        val = self.fc_val_1(val)
        value_output = self.fc_val_2(val)

        return policy_output, value_output

class PDProcess:
    def __init__(self, residual_blocks, residual_filters, batch_size):
        # Network structure
        self.residual_blocks = residual_blocks
        self.residual_filters = residual_filters
        self.model = ResNet(residual_blocks, residual_filters)
        self.batch_size = batch_size
        paddle.summary(self.model,(-1,18,19,19))

        # model type: full precision (fp32) or mixed precision (fp16)
        self.model_dtype = 'float32'

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == 'float32' else 128

        # L2 regularization parameter applied to weights.
        self.l2_scale = 1e-4

        # Set number of GPUs for training
        self.gpus_num = 1

        # For exporting
        self.weights = []

        # Output weight file with averaged weights
        self.swa_enabled = False

        # Net sampling rate (e.g 2 == every 2nd network).
        self.swa_c = 1

        # Take an exponentially weighted moving average over this
        # many networks. Under the SWA assumptions, this will reduce
        # the distance to the optimal value by a factor of 1/sqrt(n)
        self.swa_max_n = 16

        # Recalculate SWA weight batchnorm means and variances
        self.swa_recalc_bn = True

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        #config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        #self.session = tf.Session(config=config)

    def tower_loss(self, x, y_, z_):
        y_conv, z_conv = self.construct_net(x)

        # Cast the nn result back to fp32 to avoid loss overflow/underflow
        if self.model_dtype != tf.float32:
            y_conv = tf.cast(y_conv, tf.float32)
            z_conv = tf.cast(z_conv, tf.float32)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                    logits=y_conv)
        policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        mse_loss = \
            tf.reduce_mean(tf.squared_difference(z_, z_conv))

        # Regularizer
        reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
        reg_term = self.l2_scale * tf.add_n(
            [tf.cast(tf.nn.l2_loss(v), tf.float32) for v in reg_variables])

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        loss = 1.0 * policy_loss + 1.0 * mse_loss + reg_term

        return loss, policy_loss, mse_loss, reg_term, y_conv

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            if isinstance(weights, str):
                weights = tf.get_default_graph().get_tensor_by_name(weights)
            if weights.name.endswith('/batch_normalization/beta:0'):
                # Batch norm beta is written as bias before the batch
                # normalization in the weight file for backwards
                # compatibility reasons.
                bias = tf.constant(new_weights[e], shape=weights.shape)
                # Weight file order: bias, means, variances
                var = tf.constant(new_weights[e + 2], shape=weights.shape)
                new_beta = tf.divide(bias, tf.sqrt(var + tf.constant(1e-5)))
                self.assign(weights, new_beta)
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_weight)
        #This should result in identical file to the starting one
        #self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        print("Restoring from {0}".format(file))
        params = paddle.load(file)
        self.model.set_dict(params, use_structured_name=False)

    def measure_loss(self, model, batch, training=False):
        planes = np.frombuffer(batch[0], dtype=np.uint8)
        planes = np.reshape(planes,[self.batch_size,18,19,19])
        probs =  np.frombuffer(batch[1], dtype=np.float32)
        probs = np.reshape(probs,[self.batch_size,362])
        winner = np.frombuffer(batch[2], dtype=np.float32)
        winner = np.reshape(winner,[self.batch_size,1])

        planes_t = paddle.to_tensor(planes,dtype='float32')
        probs_t = paddle.to_tensor(probs,dtype='float32')
        winner_t = paddle.to_tensor(winner,dtype='float32')

        policy, value = model(planes_t)
        policy_loss = F.cross_entropy(policy, probs_t,soft_label=True)
        value_loss = F.mse_loss(value, winner_t)
        probs_max = paddle.argmax(probs_t,axis=1,keepdim=True)
        acc = paddle.metric.accuracy(policy, probs_max)
        if training:
            loss = paddle.add(policy_loss, value_loss)
            loss.backward()

        # Google's paper scales mse by 1/4 to a [0,1] range, so we do the same here
        return {'policy': policy_loss.numpy(), 'mse': value_loss.numpy()/4., 
                'accuracy':acc.numpy(), 'total': policy_loss.numpy()+value_loss.numpy()}

    def process(self, train_data, test_data):
        info_steps=50
        stats = Stats()
        timer = Timer()
        optim = paddle.optimizer.Adam(learning_rate=0.0005, parameters=self.model.parameters(),weight_decay=self.l2_scale)
        steps = 0
        while True:
            batch = next(train_data)
            # Measure losses and compute gradients for this batch.
            losses = self.measure_loss(self.model, batch, training=True)
            stats.add(losses)
            optim.step()
            optim.clear_grad()
            # fetch the current global step.

            if steps % info_steps == 0:
                speed = info_steps * self.batch_size / timer.elapsed()
                print("step {}, policy={:g} mse={:g} total={:g} ({:g} pos/s)".format(
                    steps, stats.mean('policy'), stats.mean('mse'),
                    stats.mean('total'), speed))
                stats.clear()

            if steps % 4000 == 0 and steps !=0 :
                test_stats = Stats()
                test_batches = 80 # reduce sample mean variance by ~28x
                for _ in range(0, test_batches):
                    test_batch = next(test_data)
                    losses = self.measure_loss(self.model, test_batch, training=False)
                    test_stats.add(losses)

                print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".\
                    format(steps, test_stats.mean('policy'),
                        test_stats.mean('accuracy')*100.0,
                        test_stats.mean('mse')))

                # Write out current model and checkpoint
                path = os.path.join(os.getcwd(), "leelaz-model.pdparams")
                state_dict = self.model.state_dict()
                paddle.save(state_dict, path)

                print("Model saved in file: {}".format(path))
                leela_path = path + "-" + str(steps) + ".txt"
                #self.save_leelaz_weights(leela_path)
                print("Leela weights saved to {}".format(leela_path))
                # Things have likely changed enough
                # that stats are no longer valid.

                if self.swa_enabled:
                    self.save_swa_network(steps, path, leela_path, train_data)
                print("Model saved in file: {}".format(path))

            steps += 1


    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                if weights.name.endswith('/batch_normalization/beta:0'):
                    # Batch norm beta needs to be converted to biases before
                    # the batch norm for backwards compatibility reasons
                    var_key = weights.name.replace('beta', 'moving_variance')
                    var = tf.get_default_graph().get_tensor_by_name(var_key)
                    work_weights = tf.multiply(weights,
                                               tf.sqrt(var + tf.constant(1e-5)))
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.numpy()
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def reset_batchnorm_key(self):
        self.batch_norm_count = 0
        self.reuse_var = True

    def add_weights(self, var):
        if self.reuse_var is None:
            if var.name[-11:] == "fp16_cast:0":
                name = var.name[:-12] + ":0"
                var = tf.get_default_graph().get_tensor_by_name(name)
            # All trainable variables should be stored as fp32
            assert var.dtype.base_dtype == tf.float32
            self.weights.append(var)

    def batch_norm(self, net):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        scope = self.get_batchnorm_key()
        with tf.variable_scope(scope,
                               custom_getter=float32_variable_storage_getter):
            net = tf.layers.batch_normalization(
                    net,
                    epsilon=1e-5, axis=1, fused=True,
                    center=True, scale=False,
                    training=self.training,
                    reuse=self.reuse_var)

        for v in ['beta', 'moving_mean', 'moving_variance' ]:
            name = "fp32_storage/" + scope + '/batch_normalization/' + v + ':0'
            var = tf.get_default_graph().get_tensor_by_name(name)
            self.add_weights(var)

        return net

    def conv_block(self, inputs, filter_size, input_channels, output_channels, name):
        W_conv = weight_variable(
            name,
            [filter_size, filter_size, input_channels, output_channels],
            self.model_dtype)

        self.add_weights(W_conv)

        net = inputs
        net = conv2d(net, W_conv)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        return net

    def residual_block(self, inputs, channels, name):
        net = inputs
        orig = tf.identity(net)

        # First convnet weights
        W_conv_1 = weight_variable(name + "_conv_1", [3, 3, channels, channels],
                                   self.model_dtype)
        self.add_weights(W_conv_1)

        net = conv2d(net, W_conv_1)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)

        # Second convnet weights
        W_conv_2 = weight_variable(name + "_conv_2", [3, 3, channels, channels],
                                   self.model_dtype)
        self.add_weights(W_conv_2)

        net = conv2d(net, W_conv_2)
        net = self.batch_norm(net)
        net = tf.add(net, orig)
        net = tf.nn.relu(net)

        return net

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=18,
                               output_channels=self.residual_filters,
                               name="first_conv")
        # Residual tower
        for i in range(0, self.residual_blocks):
            block_name = "res_" + str(i)
            flow = self.residual_block(flow, self.residual_filters,
                                       name=block_name)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.residual_filters,
                                   output_channels=2,
                                   name="policy_head")
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * 19 * 19])
        W_fc1 = weight_variable("w_fc_1", [2 * 19 * 19, (19 * 19) + 1], self.model_dtype)
        b_fc1 = bias_variable("b_fc_1", [(19 * 19) + 1], self.model_dtype)
        self.add_weights(W_fc1)
        self.add_weights(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.residual_filters,
                                   output_channels=1,
                                   name="value_head")
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19 * 19])
        W_fc2 = weight_variable("w_fc_2", [19 * 19, 256], self.model_dtype)
        b_fc2 = bias_variable("b_fc_2", [256], self.model_dtype)
        self.add_weights(W_fc2)
        self.add_weights(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable("w_fc_3", [256, 1], self.model_dtype)
        b_fc3 = bias_variable("b_fc_3", [1], self.model_dtype)
        self.add_weights(W_fc3)
        self.add_weights(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        return h_fc1, h_fc3

    def snap_save(self):
        # Save a snapshot of all the variables in the current graph.
        if not hasattr(self, 'save_op'):
            save_ops = []
            rest_ops = []
            for var in self.weights:
                if isinstance(var, str):
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/'+name, trainable=False)
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.restore_op)

    def save_swa_network(self, steps, path, leela_path, data):
        # Sample 1 in self.swa_c of the networks. Compute in this way so
        # that it's safe to change the value of self.swa_c
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))
        if rem > 0:
            return
        self.swa_skip.load(self.swa_c, self.session)

        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)

        if self.swa_max_n != None:
            num = min(num, self.swa_max_n)
            self.swa_count.load(float(num), self.session)

        swa_path = path + "-swa-" + str(int(num)) + "-" + str(steps) + ".txt"

        # save the current network.
        self.snap_save()
        # Copy the swa weights into the current network.
        self.session.run(self.swa_load_op)
        if self.swa_recalc_bn:
            print("Refining SWA batch normalization")
            for _ in range(200):
                batch = next(data)
                self.session.run(
                    [self.loss, self.update_ops],
                    feed_dict={self.training: True,
                               self.planes: batch[0], self.probs: batch[1],
                               self.winner: batch[2]})

        self.save_leelaz_weights(swa_path)
        # restore the saved network.
        self.snap_restore()

        print("Wrote averaged network to {}".format(swa_path))

# Unit tests for TFProcess.
def gen_block(size, f_in, f_out):
    return [ [1.1] * size * size * f_in * f_out, # conv
             [-.1] * f_out,  # bias weights
             [-.2] * f_out,  # batch norm mean
             [-.3] * f_out ] # batch norm var

class TFProcessTest(unittest.TestCase):
    def test_can_replace_weights(self):
        tfprocess = TFProcess(6, 128)
        tfprocess.init(batch_size=1)
        # use known data to test replace_weights() works.
        data = gen_block(3, 18, tfprocess.residual_filters) # input conv
        for _ in range(tfprocess.residual_blocks):
            data.extend(gen_block(3,
                tfprocess.residual_filters, tfprocess.residual_filters))
            data.extend(gen_block(3,
                tfprocess.residual_filters, tfprocess.residual_filters))
        # policy
        data.extend(gen_block(1, tfprocess.residual_filters, 2))
        data.append([0.4] * 2*19*19 * (19*19+1))
        data.append([0.5] * (19*19+1))
        # value
        data.extend(gen_block(1, tfprocess.residual_filters, 1))
        data.append([0.6] * 19*19 * 256)
        data.append([0.7] * 256)
        data.append([0.8] * 256)
        data.append([0.9] * 1)
        tfprocess.replace_weights(data)

if __name__ == '__main__':
    unittest.main()
