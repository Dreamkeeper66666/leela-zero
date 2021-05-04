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
#device = pd.set_device('gpu')
import time
import unittest
import sys
pd.disable_static()



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
from paddle import ParamAttr

# 首先实现中间两个卷积层，Skip Connection 1x1 卷积层的残差模块。代码如下：
# 残差模块

class Conv_Block(nn.Layer):
    def __init__(self, in_channel, out_channel,  kernel_size=3, stride=1, padding='SAME', relu = True):
        super(Conv_Block, self).__init__()
        
        # 第一个卷积单元
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm(out_channel,momentum=0.99,
            param_attr=ParamAttr(trainable=True),
            bias_attr=ParamAttr(trainable=True),
            use_global_stats=True)

        #self.bn1 = nn.BatchNorm2D(out_channel)
        self.relu = relu
        

    def forward(self, x):
        # 前向计算
        # [b, c, h, w], 通过第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        if self.relu:
            out = F.relu(out)

        return out

class Res_Block(nn.Layer):
    def __init__(self, in_channel, out_channel, stride=1, padding='SAME'):
        super(Res_Block, self).__init__()
        
        # 第一个卷积单元
        self.conv1 = Conv_Block(out_channel,out_channel, kernel_size=3, stride=1, padding = padding)
        #self.bn1 = nn.BatchNorm2D(out_channel)

        # 第二个卷积单元
        self.conv2 = Conv_Block(out_channel,out_channel, kernel_size=3, stride=1, padding = padding, relu=False)

        #self.bn2 = nn.BatchNorm2D(out_channel)

    def forward(self, x):
        # 前向计算
        # [b, c, h, w], 通过第一个卷积单元
        out = self.conv1(x)
        # 通过第二个卷积单元
        out = self.conv2(out)

        #  2 条路径输出直接相加,然后输入激活函数

        return out

class ResNet(nn.Layer):
    # 继承paddle.nn.Layer定义网络结构
    def __init__(self, residual_blocks=20, residual_filters=256):
        super(ResNet, self).__init__()
        # 初始化函数(根网络，预处理)
        # x:[b, c, h ,w]=[b,18,19,19]
        self.flow = Conv_Block(in_channel=18, out_channel=residual_filters, kernel_size=3, 
            stride=1, padding='SAME')# 第一层卷积,x:[b,18,256,256]
  
        self.trunk = nn.Sequential(*[
                Res_Block(residual_filters, residual_filters)
                for _ in range(residual_blocks)])
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
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


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
        val = self.relu(val)
        val = self.fc_val_2(val)
        value_output = self.tanh(val)

        return policy_output, value_output

class PDProcess:
    def __init__(self, residual_blocks, residual_filters, batch_size):
        # Network structure
        self.residual_blocks = residual_blocks
        self.residual_filters = residual_filters
        self.model = ResNet(residual_blocks, residual_filters)
        self.l2_scale = 1e-4
        #self.optim = paddle.optimizer.Momentum(learning_rate=0.005, parameters=self.model.parameters(), momentum=0.9, use_nesterov=True, weight_decay=self.l2_scale)
        self.optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=self.model.parameters(),weight_decay=self.l2_scale)
        self.batch_size = batch_size
        paddle.summary(self.model,(-1,18,19,19))

    def restore(self, file):
        print("Restoring from {0}".format(file))
        params = paddle.load(file)
        opt_state_dict = paddle.load("momentum.pdopt")
        self.model.set_state_dict(params)
        self.optim.set_state_dict(opt_state_dict)

        

    def measure_loss(self, model, batch, training=False):
        planes = np.frombuffer(batch[0], dtype=np.uint8).astype(np.float32)
        planes = np.reshape(planes,[self.batch_size,18,19,19])
        #print(planes[0][0])

        probs =  np.frombuffer(batch[1], dtype=np.float32)
        probs = np.reshape(probs,[self.batch_size,362])
        winner = np.frombuffer(batch[2], dtype=np.float32)
        winner = np.reshape(winner,[self.batch_size,1])

        planes_t = paddle.to_tensor(planes,dtype='float32')
        probs_t = paddle.to_tensor(probs,dtype='float32')
        winner_t = paddle.to_tensor(winner,dtype='float32')

        policy, value = model(planes_t)
        policy_loss = F.softmax_with_cross_entropy(logits=policy, label=probs_t, soft_label=True)
        value_loss = F.mse_loss(value, winner_t)
        probs_max = paddle.argmax(probs_t,axis=1,keepdim=True)
        acc = paddle.metric.accuracy(policy, probs_max)
        if training:
            loss = policy_loss + value_loss
            loss.backward()
        '''
        else:
            test_array = np.zeros((1,18,19,19))
            test_array[:,17,:,:] =  np.ones((1,19,19))
            test_t = paddle.to_tensor(test_array,dtype='float32')
            model.eval()
            policy, value = model(test_t)
            p = F.softmax(policy).numpy()[:,:-1].reshape(19,19)
            pb = p > 0.01
            print(np.array(pb, dtype=np.int))
            print(F.tanh(value))
        '''


        # Google's paper scales mse by 1/4 to a [0,1] range, so we do the same here
        return {'policy': policy_loss.numpy(), 'mse': value_loss.numpy()/4., 
                'accuracy':acc.numpy(), 'total': policy_loss.numpy()+value_loss.numpy()}

    def process(self, train_data, test_data):
        info_steps=50
        stats = Stats()
        timer = Timer()
        
        steps = 0
        while True:
            self.model.train()
            batch = next(train_data)
            # Measure losses and compute gradients for this batch.
            losses = self.measure_loss(self.model, batch, training=True)
            stats.add(losses)
            self.optim.step()
            self.optim.clear_grad()
            # fetch the current global step.

            if steps % info_steps == 0:
                speed = info_steps * self.batch_size / timer.elapsed()
                print("step {}, policy={:g} mse={:g} total={:g} ({:g} pos/s)".format(
                    steps, stats.mean('policy'), stats.mean('mse'),
                    stats.mean('total'), speed))
                stats.clear()

            if steps % 1000 == 0 and steps !=0 :
                # Write out current model and checkpoint
                path = os.path.join(os.getcwd(), "leelaz-model.pdparams")
                state_dict = self.model.state_dict()
                paddle.save(state_dict, path)
                paddle.save(self.optim.state_dict(), "momentum.pdopt")

                self.model.eval()

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
  

if __name__ == '__main__':
    unittest.main()
