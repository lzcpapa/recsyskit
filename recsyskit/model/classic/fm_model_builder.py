# -*- coding: utf-8 -*-
# @Author : lzcpapa 
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/12
import tensorflow as tf
from recsyskit.model.model_builder import ModelBuilder


class FMModelBuilder(ModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(FMModelBuilder, self).__init__(conf, ops_collections)

    def _fm_layer(self):
        #linear_part = self._linear_layer()
        linear_part = tf.math.reduce_sum(self._linear_layer(),
                                    axis=-1,
                                    keepdims=True)
        feature_embedding_dict = {}
        for feature_name in self.conf.feature_conf['feature_type']:
            if  self.conf.feature_conf['feature_type'][feature_name] == 'categorical':
                embedding, embedding_table, hash_tensor = self._get_feature_embedding(
                    feature_name,
                    self.conf.model_conf['default_embedding_size'],
                    self.conf.model_conf['default_hash_bucket_size'],
                    combiner='mean')

            feature_embedding_dict[feature_name] = embedding

        square_sum_embedding = tf.square(
            tf.add_n([v for (k, v) in feature_embedding_dict.items()]))
        sum_square_embedding = tf.add_n(
            [tf.square(v) for (k, v) in feature_embedding_dict.items()])

        bi_interaction = 0.5 * (square_sum_embedding - sum_square_embedding)
        bi_interaction = tf.math.reduce_sum(bi_interaction, axis=-1, keepdims=True)
        fm_logit = linear_part + bi_interaction

        return fm_logit

    def _fm_layer_v2(self):
        """
        多值特征无损
        """
        #linear_part = self._linear_layer()
        linear_part = tf.math.reduce_sum(self._linear_layer(),
                                    axis=-1,
                                    keepdims=True)

        feature_embedding_dict = {}
        feature_square_embedding_dict = {}
        for feature_name in self.conf.feature_conf['feature_type']:
            if  self.conf.feature_conf['feature_type'][feature_name] == 'categorical':
                embedding, embedding_table, hash_tensor = self._get_feature_embedding(
                    feature_name,
                    self.conf.model_conf['default_embedding_size'],
                    self.conf.model_conf['default_hash_bucket_size'],
                    combiner='sum')

                square_embedding_table = tf.square(embedding_table)
                if isinstance(hash_tensor, tf.SparseTensor):
                    squre_embedding = tf.nn.safe_embedding_lookup_sparse(
                        square_embedding_table, hash_tensor, combiner='sum')
                else:
                    squre_embedding = tf.nn.embedding_lookup(
                        square_embedding_table, hash_tensor)

                feature_embedding_dict[feature_name] = tf.squeeze(embedding)
                feature_square_embedding_dict[feature_name] = tf.squeeze(
                    squre_embedding)
        
        square_sum_embedding = tf.square(
            tf.add_n([v for (k, v) in feature_embedding_dict.items()]))
        sum_square_embedding = tf.add_n(
            [v for (k, v) in feature_square_embedding_dict.items()])
        bi_interaction = 0.5 * (square_sum_embedding - sum_square_embedding)
        bi_interaction = tf.math.reduce_sum(bi_interaction, axis=-1, keepdims=True)
        fm_logit = linear_part + bi_interaction
        return fm_logit

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))
        output = self._fm_layer_v2()
        output = tf.nn.sigmoid(output)

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
