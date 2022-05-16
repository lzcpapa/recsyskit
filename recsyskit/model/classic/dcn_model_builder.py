# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/20
"""DCN Model"""

import tensorflow as tf
from recsyskit.model import layers
from recsyskit.model.model_builder import ModelBuilder


class DCNModelBuilder(ModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(DCNModelBuilder, self).__init__(conf, ops_collections)

    def _cross_network(self, input, cross_layer_num=3):
        input_dim = tf.shape(input)[-1]
        w_initial_value = tf.random.truncated_normal(shape=(input_dim, ),
                                                     stddev=0.01)
        b_initial_value = tf.zeros(shape=(input_dim, ))
        x_0 = input
        x_l = input
        for i in range(cross_layer_num):
            W = tf.Variable(w_initial_value, name='cross_layer_w{}'.format(i))
            b = tf.Variable(b_initial_value, name='cross_layer_b{}'.format(i))
            tmp = tf.tensordot(tf.reshape(x_l, (-1, 1, input_dim)), W,
                               axes=1)  # tf.einsum('ijk,k->ij', a,b)
            x_l = x_0 * tmp + b + x_l

        return x_l

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))

        input_layer = self._get_embedding_input_layer(
            self.conf.model_conf['default_embedding_size'],
            self.conf.model_conf['default_hash_bucket_size'])
        input_layer = tf.concat(input_layer, -1)

        with tf.compat.v1.variable_scope('cross_network') as scope:
            cross_output = self._cross_network(input_layer)

        with tf.compat.v1.variable_scope('deep_network') as scope:
            dnn_output = self._mlp_layer(
                input_layer,
                layers=[1024, 256, 128],
                activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu])

        output = self._mlp_layer(tf.concat([cross_output, dnn_output],
                                               axis=-1),
                                     layers=[1],
                                     activations=[tf.nn.sigmoid])

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
