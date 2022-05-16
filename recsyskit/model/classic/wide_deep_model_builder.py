# -*- coding: utf-8 -*-
# @Author : lzcpapa 
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/13
import tensorflow as tf
from recsyskit.model.model_builder import ModelBuilder


class WideDeepModelBuilder(ModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(WideDeepModelBuilder, self).__init__(conf, ops_collections)

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))

        with tf.compat.v1.variable_scope('wide') as scope:
            self.ops_collections['wide_scope'] = scope.name
            print(self.ops_collections['wide_scope'])
            linear_out = tf.math.reduce_sum(self._linear_layer(),
                                       axis=-1,
                                       keepdims=True)

        with tf.compat.v1.variable_scope('deep') as scope:
            self.ops_collections['deep_scope'] = scope.name
            print(self.ops_collections['deep_scope'])
            input_layer = self._get_embedding_input_layer(
                self.conf.model_conf['default_embedding_size'],
                self.conf.model_conf['default_hash_bucket_size'])
            input_layer = tf.concat(input_layer, -1)

            dnn_output = self._mlp_layer(
                input_layer,
                layers=[1024, 256, 1],
                activations=[tf.nn.relu, tf.nn.relu, None])

        output = tf.nn.sigmoid(linear_out + dnn_output)

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()

    def _build_loss(self, logit):
        self.ops_collections['step'] = tf.compat.v1.train.get_global_step()
        cross_entropy = tf.compat.v1.losses.log_loss(
            labels=self.ops_collections['label'], predictions=logit)
        losses = tf.compat.v1.reduce_mean(cross_entropy)
        losses += tf.compat.v1.losses.get_regularization_loss()

        wide_optimizer = tf.compat.v1.train.FtrlOptimizer(0.001)
        wide_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.ops_collections['wide_scope'])
        print(wide_vars)
        wide_train_op = wide_optimizer.minimize(losses, var_list=wide_vars)

        deep_optimizer = tf.compat.v1.train.AdamOptimizer(0.001)

        deep_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.ops_collections['deep_scope'])
        deep_train_op = deep_optimizer.minimize(
            losses, global_step=self.ops_collections['step'], var_list=deep_vars)

        self.ops_collections['loss'] = losses
        self.ops_collections['train_op'] = tf.compat.v1.group(
            wide_train_op, deep_train_op)
