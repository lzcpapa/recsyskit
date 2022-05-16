# -*- coding: utf-8 -*-
# @Author : lzcpapa
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/12

import json
from collections import namedtuple

import tensorflow as tf

from recsyskit.model.classic.dcnv2_model_builder import DCNV2ModelBuilder
from recsyskit.model.classic.dcn_model_builder import DCNModelBuilder
from recsyskit.model.classic.wide_deep_model_builder import WideDeepModelBuilder
from recsyskit.model.classic.biffm_model_builder import BiFFMModelBuilder
from recsyskit.model.classic.deepfm_model_builder import DeepFMModelBuilder
from recsyskit.model.classic.ffm_model_builder import FFMModelBuilder
from recsyskit.model.classic.fm_model_builder import FMModelBuilder
from recsyskit.model.input_builder import ExampleInputBuilder
from recsyskit.model.model_builder import ModelBuilder
from recsyskit.model.classic.nfm_model_builder import NFMModelBuilder
from recsyskit.utils.feature_config import FeatureConfig

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string('config', 'conf.json', 'config')
tf.compat.v1.app.flags.DEFINE_boolean('do_train', False, 'train')
tf.compat.v1.app.flags.DEFINE_boolean('do_eval', False, 'eval')


def get_config():
    with open(FLAGS.config, 'r') as mc:
        config = json.load(mc)
    config['feature_conf'] = FeatureConfig.get_feature_config(config['run_conf']['dataset'])
    Config = namedtuple("Config", config)
    return Config(**config)


def get_model(conf):
    if conf.model_conf['model'] == 'FM':
        return FMModelBuilder(conf)
    elif conf.model_conf['model'] == 'DNN':
        return ModelBuilder(conf)
    elif conf.model_conf['model'] == 'FFM':
        return FFMModelBuilder(conf)
    elif conf.model_conf['model'] == 'BiFFM':
        return BiFFMModelBuilder(conf)
    elif conf.model_conf['model'] == 'NFM':
        return NFMModelBuilder(conf)
    elif conf.model_conf['model'] == 'DeepFM':
        return DeepFMModelBuilder(conf)
    elif conf.model_conf['model'] == 'WideDeep':
        return WideDeepModelBuilder(conf)
    elif conf.model_conf['model'] == 'DCN':
        return DCNModelBuilder(conf)
    elif conf.model_conf['model'] == 'xDeepFM':
        raise NotImplementedError
    elif conf.model_conf['model'] == 'DCN-V2':
        return DCNV2ModelBuilder(conf)
    else:
        raise NotImplementedError


def main(argv):

    config = get_config()

    if not tf.compat.v1.gfile.Exists(config.run_conf['checkpoint']):
        tf.compat.v1.gfile.MkDir(config.run_conf['checkpoint'])

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=config.run_conf['checkpoint'],
        session_config=sess_config,
        #saver_max_to_keep=FLAGS.saver_max_to_keep,
        #save_summary_steps=FLAGS.save_summary_steps,
        #save_checkpoints_steps = args.check_point_num if hvd.rank() == 0 else None,
        #log_step_count_steps = args.log_every_n_steps if hvd.rank() == 0 else None
        save_checkpoints_steps=config.run_conf['save_checkpoints_steps'])

    input_builder = ExampleInputBuilder(config.feature_conf['feature_spec'],
                                        config)
    print(config.model_conf)
    model = get_model(config)
    model_fn = model.model_fn_builder()
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    if FLAGS.do_train:
        train_input_fn = input_builder.input_fn_builder(
            config.run_conf['train_data_path'],
            epoch=config.model_conf['epoch'],
            compression_type='GZIP')

        estimator.train(input_fn=train_input_fn, hooks=[], max_steps=10000000)

    if FLAGS.do_eval:
        eval_input_fn = input_builder.input_fn_builder(
            config.run_conf['test_data_path'], compression_type='GZIP')
        result = estimator.evaluate(input_fn=eval_input_fn)


if __name__ == '__main__':
    tf.compat.v1.app.run()
