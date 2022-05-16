# -*- coding: utf-8 -*-
# @Author : lzcpapa 
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/12
import tensorflow as tf
from recsyskit.model.model_builder import ModelBuilder


class NFMModelBuilder(ModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(NFMModelBuilder, self).__init__(conf, ops_collections)

    def _bi_interaction_layer(self, input):
        sum_v = tf.add_n(input)
        suqare_v = tf.square(input)
        return tf.square(sum_v) - tf.einsum('fbk->bk', suqare_v)

    def _nfm_layer(self):
        #linear_part = self._linear_layer()
        linear_part = tf.math.reduce_sum(self._linear_layer(), axis=-1, keepdims=True)
        input_layer = self._get_embedding_input_layer(
            self.conf.model_conf['default_embedding_size'],
            self.conf.model_conf['default_hash_bucket_size'])

        bi_interaction = self._bi_interaction_layer(input_layer)
        return linear_part + self._mlp_layer(
            bi_interaction,
            layers=[128, 32, 1],
            activations=[None, None, None])

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))

        output = tf.nn.sigmoid(self._nfm_layer())

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
