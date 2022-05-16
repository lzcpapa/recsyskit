# -*- coding: utf-8 -*-
# @Author : lzcpapa 
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/09
import tensorflow as tf
from recsyskit.utils.hook import DatasetInitializerHook


class InputBuilder:
    def __init__(self, feature_spec={}, conf=None):
        self.conf = conf
        self.feature_spec = feature_spec

    def _parse_proto(self, example_proto):
        return tf.compat.v1.io.parse_example(example_proto, self.feature_spec)

    def _dataset_func(self, input, compression_type=None):
        return tf.compat.v1.data.TFRecordDataset(input,
                                       compression_type=compression_type)

    def _get_dataset(self, input_path, **kwargs):
        tf.compat.v1.logging.info('Feature spec info: {}'.format(self.feature_spec))
        input_files = tf.compat.v1.data.Dataset.list_files(input_path, shuffle=True)
        dataset = input_files.interleave(
            lambda x: self._dataset_func(
                x, kwargs.get('compression_type', None)),
            cycle_length=tf.compat.v1.data.experimental.AUTOTUNE,
            block_length=10,
            num_parallel_calls=tf.compat.v1.data.experimental.AUTOTUNE)
        #files = tf.compat.v1.data.Dataset.list_files(files)
        dataset = dataset.repeat(kwargs.get('epoch', 1))
        dataset = dataset.batch(self.conf.model_conf['batch_size'],
                                drop_remainder=True)
        dataset = dataset.map(self._parse_proto,
                              num_parallel_calls=tf.compat.v1.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.compat.v1.data.experimental.AUTOTUNE)
        return dataset

    def input_fn_builder(self,
                         input_path,
                         **kwargs):
        def input_fn():
            return self._get_dataset(input_path, **kwargs)

        return input_fn


class ExampleInputBuilder(InputBuilder):
    def __init__(self, feature_spec={}, conf=None):
        super(ExampleInputBuilder, self).__init__(feature_spec, conf)

    def _parse_proto(self, example_proto):
        return tf.compat.v1.io.parse_example(example_proto, self.feature_spec)

    def _dataset_func(self, input, compression_type=None):
        return tf.compat.v1.data.TFRecordDataset(input,
                                       compression_type=compression_type)

    def input_fn_builder(self, input_path, **kwargs):
        def input_fn():
            return self._get_dataset(input_path, **kwargs)

        return input_fn
