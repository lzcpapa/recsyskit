# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/15

import tensorflow as tf
from recsyskit.model.classic.fm_model_builder import FMModelBuilder
from itertools import combinations


class BiFFMModelBuilder(FMModelBuilder):

    def __init__(self, conf=None, ops_collections={}):
        super(BiFFMModelBuilder, self).__init__(conf, ops_collections)
        self.field_size = 3

    def _get_bilinear_weight(self,
                             weight_shape,
                             weight_name='',
                             bi_type='Field-All',
                             reuse=tf.compat.v1.AUTO_REUSE):
        shape_str = 'x'.join([str(i) for i in weight_shape])
        with tf.compat.v1.variable_scope('bilinear_weight', reuse=reuse):
            bilinear_weight = tf.compat.v1.get_variable(
                "{}_{}".format(weight_name, shape_str),
                shape=weight_shape,
                dtype=tf.float32,
                initializer=self.embedding_initializer)

            return bilinear_weight

    def _biffm_layer(self, bi_type='Field-All'):
        linear_part = tf.math.reduce_sum(self._linear_layer(),
                                         axis=-1,
                                         keepdims=True)
        feature_embedding_dict = {}
        feature_field_dict = {}
        field_size = self.field_size
        embedding_size = self.conf.model_conf['default_embedding_size']
        if bi_type == 'Field-All':
            bi_weight_shape = [embedding_size, embedding_size]
        elif bi_type == 'Field-Each':
            bi_weight_shape = [field_size, embedding_size, embedding_size]
        elif bi_type == 'Field-Interaction':
            bi_weight_shape = [
                field_size, field_size, embedding_size, embedding_size
            ]
        bilinear_weight = self._get_bilinear_weight(
            weight_shape=bi_weight_shape, weight_name='bilinear_weight')

        for feature_name in self.conf.feature_conf['feature_type']:
            if self.conf.feature_conf['feature_type'][
                    feature_name] == 'categorical':
                embedding, embedding_table, hash_tensor = self._get_feature_embedding(
                    feature_name,
                    self.conf.model_conf['default_embedding_size'],
                    self.conf.model_conf['default_hash_bucket_size'],
                    combiner='mean')

                feature_field_dict[feature_name] = hash(
                    feature_name) % self.field_size
                print('Feature {} belong to field {}'.format(
                    feature_name, feature_field_dict[feature_name]))
                feature_embedding_dict[feature_name] = embedding

        interation_pair = combinations(feature_embedding_dict.keys(), 2)
        if bi_type == 'Field-All':
            second_order_part = [
                tf.einsum(
                    'ij,ij->i',
                    tf.einsum('ij,jj->ij', feature_embedding_dict[i],
                              bilinear_weight), feature_embedding_dict[j])
                for i, j in interation_pair
            ]
        elif bi_type == 'Field-Each':
            second_order_part = [
                tf.einsum(
                    'ij,ij->i',
                    tf.einsum('ij,jj->ij', feature_embedding_dict[i],
                              bilinear_weight[feature_field_dict[i]]),
                    feature_embedding_dict[j]) for i, j in interation_pair
            ]
        elif bi_type == 'Field-Interaction':
            second_order_part = [
                tf.einsum(
                    'ij,ij->i',
                    tf.einsum('ij,jj->ij', feature_embedding_dict[i],
                              bilinear_weight), feature_embedding_dict[j])
                for i, j in interation_pair
            ]
        else:
            raise ValueError()

        second_order_part = tf.reshape(tf.add_n(second_order_part), (-1, 1))  #
        print("second part:{}".format(second_order_part))
        return linear_part + second_order_part

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))
        output = self._biffm_layer(bi_type='Field-Each')
        output = tf.nn.sigmoid(output)

        self.ops_collections['pred'] = output
        self._build_loss(output)
        self._build_metric()
        self._debug_and_summary()
