# -*- coding: utf-8 -*-
# @Author : lzcpapa 
# @Email  : miningsecret@gmail.com
# @Time   : 2022/04/09
import tensorflow as tf

class DatasetInitializerHook(tf.compat.v1.train.SessionRunHook):
  """Creates a SessionRunHook that initializes the passed iterator."""
  def __init__(self):
      self.initializer = None
      super(DatasetInitializerHook, self).__init__()


  def after_create_session(self, session, coord):
      session.run(self.initializer)