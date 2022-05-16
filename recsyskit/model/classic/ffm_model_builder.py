# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/14
import tensorflow as tf
from recsyskit.model.classic.fm_model_builder import FMModelBuilder
from itertools import combinations


class FFMModelBuilder(FMModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(FFMModelBuilder, self).__init__(conf, ops_collections)
        self.field_size = 2

    def _get_feature_fields_embedding(self,
                                      feature_name,
                                      emb_size,
                                      hash_bucket_size,
                                      filed_size,
                                      combiner='sum',
                                      feature_table_name='',
                                      zero_padding=False,
                                      reuse=tf.compat.v1.AUTO_REUSE):
        feature_input = self.ops_collections[feature_name]
        feature_table_name = '{}_embedding_table'.format(
            feature_name) if feature_table_name == '' else feature_table_name
        with tf.compat.v1.variable_scope('feature_embedding_table',
                                         reuse=reuse):
            embedding_table = tf.compat.v1.get_variable(
                "{}_{}x{}x{}".format(feature_table_name, filed_size,
                                     hash_bucket_size, emb_size),
                shape=[hash_bucket_size, filed_size, emb_size],
                dtype=tf.float32,
                initializer=self.embedding_initializer)

            if zero_padding:
                embedding_table = tf.concat(
                    (tf.zeros(shape=[1, filed_size, emb_size]),
                     embedding_table[1:, :, :]),
                    axis=0)

            hash_tensor = self._get_hash_tensor(feature_input, feature_name,
                                                hash_bucket_size)
            if isinstance(feature_input, tf.SparseTensor):
                hash_tensor = tf.sparse.to_dense(hash_tensor, default_value=0)

            embedding = tf.math.reduce_mean(
                tf.nn.embedding_lookup(embedding_table, hash_tensor),
                axis=1)  # Batch * Field * Embedding
            return embedding, embedding_table, hash_tensor

    def _ffm_layer(self):
        #linear_part = self._linear_layer()
        linear_part = tf.math.reduce_sum(self._linear_layer(),
                                         axis=-1,
                                         keepdims=True)
        feature_embedding_dict = {}
        feature_field_dict = {}
        for feature_name in self.conf.feature_conf['feature_type']:
            if self.conf.feature_conf['feature_type'][
                    feature_name] == 'categorical':
                embedding, embedding_table, hash_tensor = self._get_feature_fields_embedding(
                    feature_name,
                    self.conf.model_conf['default_embedding_size'],
                    self.conf.model_conf['default_hash_bucket_size'],
                    filed_size=self.field_size,
                    combiner='mean')

                feature_field_dict[feature_name] = hash(
                    feature_name) % self.field_size
                print('Feature {} belong to field {}'.format(
                    feature_name, feature_field_dict[feature_name]))
                feature_embedding_dict[feature_name] = embedding

        second_order_part = []
        for i, j in combinations(feature_embedding_dict.keys(), 2):
            fi = feature_field_dict[i]
            fj = feature_field_dict[j]
            # batch * k
            vv = tf.multiply(feature_embedding_dict[i][:, fj, :],
                             feature_embedding_dict[j][:, fi, :])

            second_order_part.append(vv)
        second_order_part = tf.math.reduce_sum(tf.add_n(second_order_part),
                                               axis=-1,
                                               keepdims=True)
        return linear_part + second_order_part

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))
        output = self._ffm_layer()
        output = tf.nn.sigmoid(output)

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
