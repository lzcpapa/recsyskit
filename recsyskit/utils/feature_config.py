# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/09

import tensorflow as tf


class FeatureConfig:

    AVAZU_FEATURE_TYPE = {
        'C1': 'categorical',
        'C14': 'categorical',
        'C15': 'categorical',
        'C16': 'categorical',
        'C17': 'categorical',
        'C18': 'categorical',
        'C19': 'categorical',
        'C20': 'categorical',
        'C21': 'categorical',
        'banner_pos': 'categorical',
        'site_id': 'categorical',
        'site_domain': 'categorical',
        'site_categorical': 'categorical',
        'app_id': 'categorical',
        'app_domain': 'categorical',
        'app_categorical': 'categorical',
        'device_id': 'categorical',
        'device_ip': 'categorical',
        'device_model': 'categorical',
        'device_type': 'categorical',
        'device_conn_type': 'categorical',
    }

    AVAZU_FEATURE_SPEC = {
        'C1':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C14':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C15':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C16':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C17':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C18':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C19':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C20':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C21':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'banner_pos':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'site_id':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'site_domain':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'site_categorical':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'app_id':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'app_domain':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'app_categorical':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'device_id':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'device_ip':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'device_model':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'device_type':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'device_conn_type':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'label':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=[0.0])
    }

    CRITEO_FEATURE_TYPE = {
        'I1': 'numerical',
        'I2': 'numerical',
        'I3': 'numerical',
        'I4': 'numerical',
        'I5': 'numerical',
        'I6': 'numerical',
        'I7': 'numerical',
        'I8': 'numerical',
        'I9': 'numerical',
        'I10': 'numerical',
        'I11': 'numerical',
        'I12': 'numerical',
        'I13': 'numerical',
        'C1': 'categorical',
        'C2': 'categorical',
        'C3': 'categorical',
        'C4': 'categorical',
        'C5': 'categorical',
        'C6': 'categorical',
        'C7': 'categorical',
        'C8': 'categorical',
        'C9': 'categorical',
        'C10': 'categorical',
        'C11': 'categorical',
        'C12': 'categorical',
        'C13': 'categorical',
        'C14': 'categorical',
        'C15': 'categorical',
        'C16': 'categorical',
        'C17': 'categorical',
        'C18': 'categorical',
        'C19': 'categorical',
        'C20': 'categorical',
        'C21': 'categorical',
        'C22': 'categorical',
        'C23': 'categorical',
        'C24': 'categorical',
        'C25': 'categorical',
        'C26': 'categorical'
    }

    CRITEO_FEATURE_SPEC = {
        'label':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I1':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I2':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I3':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I4':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I5':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I6':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I7':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I8':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I9':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I10':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I11':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I12':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'I13':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32,
                              default_value=[0.0]),
        'C1':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C2':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C3':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C4':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C5':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C6':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C7':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C8':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C9':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C10':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C11':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C12':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C13':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C14':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C15':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C16':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C17':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C18':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C19':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C20':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C21':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C22':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C23':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C24':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C25':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=['']),
        'C26':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.string, default_value=[''])
    }

    @classmethod
    def get_feature_config(cls, dataset):
        if dataset == 'avazu':
            return {
                'feature_type': cls.AVAZU_FEATURE_TYPE,
                'feature_spec': cls.AVAZU_FEATURE_SPEC
            }
        elif dataset == 'criteo':
            return {
                'feature_type': cls.CRITEO_FEATURE_TYPE,
                'feature_spec': cls.CRITEO_FEATURE_SPEC
            }
        else:
            raise NotImplementedError