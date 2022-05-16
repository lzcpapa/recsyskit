# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/13
import tensorflow as tf
from recsyskit.model.classic.fm_model_builder import FMModelBuilder


class DeepFMModelBuilder(FMModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(DeepFMModelBuilder, self).__init__(conf, ops_collections)

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))
        fm_output = self._fm_layer_v2()

        input_layer = self._get_embedding_input_layer(
            self.conf.model_conf['default_embedding_size'],
            self.conf.model_conf['default_hash_bucket_size'])
        input_layer = tf.concat(input_layer, -1)

        dnn_output = self._mlp_layer(
            input_layer,
            layers=[1024, 256, 1],
            activations=[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid])

        output = tf.nn.sigmoid(fm_output + dnn_output)

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
