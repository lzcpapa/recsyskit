# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/25
"""DCN Model"""

import tensorflow as tf
from recsyskit.model import layers
from recsyskit.model.model_builder import ModelBuilder


class DCNV2ModelBuilder(ModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(DCNV2ModelBuilder, self).__init__(conf, ops_collections)

    def _cross_network(self, input, cross_layer_num=3):
        """Normal cross network of DCN-V2

        Args:
            input (_type_): embedding layer, [batch, dim]
            cross_layer_num (int, optional): cross layer. Defaults to 3.

        Returns:
            _type_: _description_
        """
        _, input_dim = tf.shape(input)
        w_initial_value = tf.random.truncated_normal(shape=(input_dim,
                                                            input_dim),
                                                     stddev=0.01)
        b_initial_value = tf.zeros(shape=(input_dim, ))
        x_0 = input
        x_l = input  # [batch, dim]
        for i in range(cross_layer_num):
            W = tf.Variable(w_initial_value, name='cross_layer_w{}'.format(i))
            b = tf.Variable(b_initial_value, name='cross_layer_b{}'.format(i))
            tmp = tf.einsum('ik,kk->ik', x_l, W) + b  # [batch, dim]
            x_l = x_0 * tmp + x_l

        return x_l

    def _cost_effective_mixture_cross_network(self,
                                              input,
                                              subspace_experts,
                                              cross_layer_num=3):
        """Cost-Effective Mixture of Low-Rank DCN-V2

        Args:
            input (_type_): _description_
            subspace_experts (_type_): _description_
            cross_layer_num (int, optional): _description_. Defaults to 3.
        """
        x_0 = input
        x_l = input
        input_dim = tf.shape(input)[-1]
        gates_output = tf.compat.v1.layers.dense(input,
                                                 units=len(subspace_experts),
                                                 activation=tf.nn.sigmoid,
                                                 name='gate')
        for i in range(cross_layer_num):
            for j, subsparce in enumerate(subspace_experts):
                
                pass

    def _cost_effective_cross_network(self,
                                      x_0,
                                      x_l,
                                      expert_index=0,
                                      subspace_dim=0,
                                      layer_index=0):
        """Low-Rank DCN-V2

        Args:
            input (_type_): _description_
            subspace_dim (_type_, optional): _description_. Defaults to None.
            cross_layer_index (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        input_dim = tf.shape(x_0)[-1]
        low_rank_dim = int(input_dim /
                           4) if subspace_dim == 0 else int(subspace_dim)
        UV_initial_value = tf.random.truncated_normal(shape=(input_dim,
                                                             low_rank_dim),
                                                      stddev=0.01)
        b_initial_value = tf.zeros(shape=(input_dim, ))
        name = 'cross_layer_{}_expert{}'.format(layer_index, expert_index)
        U_l = tf.Variable(UV_initial_value, name='{}_u'.format(name))
        V_l = tf.Variable(UV_initial_value, name='{}_v'.format(name))
        b_l = tf.Variable(b_initial_value, name='{}_b'.format(name))
        project_r = tf.einsum('dr,bd->rb', V_l, x_l)
        project_d = tf.einsum('dr,rb->bd', U_l, project_r)

        return x_0 * (project_d + b_l)

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))
        dcn_type = 'Parallel'
        self.ops_collections['deep_scope'] = scope.name
        print(self.ops_collections['deep_scope'])
        input_layer = self._get_embedding_input_layer(
            self.conf.model_conf['default_embedding_size'],
            self.conf.model_conf['default_hash_bucket_size'])
        input_layer = tf.concat(input_layer, -1)

        with tf.compat.v1.variable_scope('cross_network') as scope:
            cross_output = self._cross_network(input_layer)

        if dcn_type == 'Parallel':
            with tf.compat.v1.variable_scope('deep_network') as scope:
                dnn_output = self._mlp_layer(
                    input_layer,
                    layers=[1024, 256, 128],
                    activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu])
            output = self._mlp_layer(cross_output + dnn_output,
                                         layers=[1],
                                         activations=[tf.nn.sigmoid])
        elif dcn_type == 'Stacked':
            with tf.compat.v1.variable_scope('deep_network') as scope:
                dnn_output = self._mlp_layer(cross_output,
                                                 layers=[1024, 256, 128, 1],
                                                 activations=[
                                                     tf.nn.relu, tf.nn.relu,
                                                     tf.nn.relu, tf.nn.sigmoid
                                                 ])
            output = dnn_output

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
