# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/09
import tensorflow as tf


class ModelBuilder:

    def __init__(self, conf=None, ops_collections={}):
        self.ops_collections = ops_collections
        self.conf = conf
        self.embedding_initializer = tf.random_normal_initializer(stddev=0.1)

    def _get_combiner(self, combiner='sum'):
        if combiner == 'sum':
            return tf.math.reduce_sum
        elif combiner == 'mean':
            return tf.math.reduce_mean
        elif combiner == 'max':
            return tf.math.reduce_max
        else:
            return None

    def _get_hash_tensor(self, input, feature_name, bucket_size):
        if input.dtype != tf.string:
            input = tf.strings.as_string(input)
        if isinstance(input, tf.SparseTensor):
            values = tf.strings.to_hash_bucket_strong(
                input=input.values,
                num_buckets=bucket_size,
                key=[0] * 2,
                name="hash_{}".format(feature_name))
            hash_sparse_tensor = tf.sparse.SparseTensor(
                indices=input.indices,
                values=values,
                dense_shape=input.dense_shape)
            return hash_sparse_tensor
        else:
            hash_tensor = tf.strings.to_hash_bucket_strong(
                input=input,
                num_buckets=bucket_size,
                key=[0] * 2,
                name="hash_{}".format(feature_name))
            return hash_tensor

    def _get_feature_embedding(self,
                               feature_name,
                               emb_size,
                               hash_bucket_size,
                               combiner='mean',
                               feature_table_name='',
                               zero_padding=False,
                               reuse=tf.compat.v1.AUTO_REUSE):
        feature_input = self.ops_collections[feature_name]
        feature_table_name = '{}_embedding_table'.format(
            feature_name) if feature_table_name == '' else feature_table_name

        with tf.compat.v1.variable_scope('feature_embedding_table',
                                         reuse=reuse):
            embedding_table = tf.compat.v1.get_variable(
                "{}_{}x{}".format(feature_table_name, hash_bucket_size,
                                  emb_size),
                shape=[hash_bucket_size, emb_size],
                dtype=tf.float32,
                initializer=self.embedding_initializer)
        
            if zero_padding:
                embedding_table = tf.concat(
                    (tf.zeros(shape=[1, emb_size]), embedding_table[1:, :]), 0)

            hash_tensor = self._get_hash_tensor(feature_input, feature_name,
                                                hash_bucket_size)
            if isinstance(feature_input, tf.SparseTensor):
                embedding = tf.nn.safe_embedding_lookup_sparse(
                    embedding_table, hash_tensor, combiner=combiner)
            else:
                embedding = tf.math.reduce_mean(tf.nn.embedding_lookup(
                    embedding_table, hash_tensor),
                                                axis=-2)
            return embedding, embedding_table, hash_tensor

    def _get_embedding_input_layer(self,
                                   embedding_size,
                                   bucket_size,
                                   combiner='mean'):
        tf.compat.v1.logging.info('Input {}'.format(self.ops_collections))

        feature_embedding_dict = {}
        for feature_name in self.conf.feature_conf['feature_type']:
            if self.conf.feature_conf['feature_type'][feature_name] == 'categorical':
                embedding, _, _ = self._get_feature_embedding(
                    feature_name, embedding_size, bucket_size, combiner)
                feature_embedding_dict[feature_name] = embedding

        feature_embedding_list = [
            v for (k, v) in feature_embedding_dict.items()
        ]
        
        tf.compat.v1.logging.info(
            'Feature inputs: {}'.format(feature_embedding_list))
        
        return feature_embedding_list

    def _debug_and_summary(self):
        self.ops_collections['_debug_ops'] = {
            'loss': self.ops_collections['loss'],
            'auc_batch': self.ops_collections['auc_batch'],
            'auc': self.ops_collections['auc'],
            'step': self.ops_collections['step'],
        }
        tf.compat.v1.summary.scalar('auc', self.ops_collections['auc'])

    def _linear_layer(self, intercept=True):

        linear_output = self._get_embedding_input_layer(
            1, self.conf.model_conf['default_hash_bucket_size'])
        linear_output = tf.concat(linear_output, -1)

        w0 = tf.compat.v1.get_variable('w0',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer,
                                       trainable=intercept)

        w0 = tf.reshape(w0 * tf.ones_like(linear_output[:, 0]), (-1, 1))

        output = tf.concat([w0, linear_output], axis=-1)

        return output

    def _mlp_layer(self,
                       input,
                       layers=[],
                       activations=[],
                       bath_norm=[],
                       name=''):
        layer_cnt = 0
        output = input
        for i, layer in enumerate(layers):
            output = tf.compat.v1.layers.dense(
                inputs=output,
                units=layer,
                activation=activations[i],
                kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.003))

        return output

    def _build_network(self, is_trainning=True):
        tf.compat.v1.logging.info(
            'Building network, is_trainning: {}'.format(is_trainning))

        input_layer = self._get_embedding_input_layer(
            self.conf.model_conf['default_embedding_size'],
            self.conf.model_conf['default_hash_bucket_size'])
        input_layer = tf.concat(input_layer, -1)

        output = self._mlp_layer(
            input_layer,
            layers=[1024, 256, 1],
            activations=[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid])

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

        self.ops_collections['loss'] = losses

        optimizer = tf.compat.v1.train.AdamOptimizer(
            self.conf.model_conf['learning_rate'], epsilon=1e-6)

        update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.compat.v1.control_dependencies(
                update_ops), tf.compat.v1.variable_scope(
                    "train", reuse=tf.compat.v1.AUTO_REUSE):
            train_op = optimizer.minimize(
                self.ops_collections['loss'],
                global_step=self.ops_collections['step'])

            self.ops_collections['train_op'] = train_op

    def _build_metric(self):
        auc_metric = tf.compat.v1.metrics.auc(
            labels=self.ops_collections['label'],
            predictions=self.ops_collections['pred'])
        self.ops_collections['auc'] = auc_metric[1]
        self.ops_collections['auc_batch'] = auc_metric[0]

    def model_fn_builder(self):
        """Provide model_fn for estimator."""

        def model_fn(features, labels, mode, params):
            self.ops_collections.update(features)
            if 'click' in self.ops_collections:
                self.ops_collections['label'] = self.ops_collections['click']

            is_training = (mode == tf.compat.v1.estimator.ModeKeys.TRAIN)
            self._build_network(is_training)
            eval_metric_ops = {
                'auc': (self.ops_collections['auc_batch'],
                        self.ops_collections['auc'])
            }
            if mode == tf.compat.v1.estimator.ModeKeys.EVAL:
                return tf.compat.v1.estimator.EstimatorSpec(
                    mode=mode,
                    loss=self.ops_collections['loss'],
                    eval_metric_ops=eval_metric_ops)

            return tf.compat.v1.estimator.EstimatorSpec(
                mode,
                loss=self.ops_collections['loss'],
                train_op=self.ops_collections['train_op'],
                training_hooks=[
                    tf.compat.v1.train.LoggingTensorHook(
                        self.ops_collections['_debug_ops'],
                        every_n_iter=self.conf.run_conf['log_steps'])
                ])

        return model_fn
