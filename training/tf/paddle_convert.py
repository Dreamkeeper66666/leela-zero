#!/usr/bin/env python3
import numpy as np
import sys
import paddle

state = paddle.load(sys.argv[1])


def tensor_to_str(t):
    return ' '.join(map(str, np.array(t).flatten()))

def convert_block(t, name, i):
    weight = np.array(t[name + '.conv{}.weight'.format(i)])
    bias = np.array(t[name + '.conv{}.bias'.format(i)])
    bn_gamma = np.array(t[name + '.bn{}.weight'.format(i)])
    bn_beta = np.array(t[name + '.bn{}.bias'.format(i)])
    bn_mean = np.array(t[name + '.bn{}._mean'.format(i)])
    bn_var = np.array(t[name + '.bn{}._variance'.format(i)])

    # y1 = weight * x + bias
    # y2 = gamma * (y1 - mean) / sqrt(var + e) + beta

    # convolution: [out, in, x, y]

    weight *= bn_gamma[:, np.newaxis, np.newaxis, np.newaxis]

    bias = bn_gamma * bias + bn_beta * np.sqrt(bn_var + 1e-5)

    bn_mean *= bn_gamma

    return [weight, bias, bn_mean, bn_var]

def write_block(f, b):
    for w in b:
        f.write(' '.join(map(str, w.flatten())) + '\n')

if 0:
    for key in state.keys():
        print(key, state[key].shape)

with open('paddle_converted_weights.txt', 'w') as f:
    # version 2 means value head is for black, not for side to move
    f.write('2\n')


    b = convert_block(state, 'flow' , 1)

    # Permutate input planes
    #p = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]
    #b[0] = b[0][:,p,:,:]

    write_block(f, b)
    for block in range(2):
        b = convert_block(state, 'trunk.{}'.format(block), 1)
        write_block(f, b)
        b = convert_block(state, 'trunk.{}'.format(block), 2)
        write_block(f, b)
    b = convert_block(state, 'conv_pol', 1)
    write_block(f, b)
    f.write(tensor_to_str(state['fc_pol.weight']) + '\n')
    f.write(tensor_to_str(state['fc_pol.bias']) + '\n')
    b = convert_block(state, 'conv_val', 1)
    write_block(f, b)
    f.write(tensor_to_str(state['fc_val_1.weight']) + '\n')
    f.write(tensor_to_str(state['fc_val_1.bias']) + '\n')
    f.write(tensor_to_str(state['fc_val_2.weight']) + '\n')
    f.write(tensor_to_str(state['fc_val_2.bias']) + '\n')
