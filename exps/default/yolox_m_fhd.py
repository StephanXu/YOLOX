#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.input_size = (1088, 1920)
        # self.test_size = (1088, 1920)
        # self.input_size = (736, 1280)
        # self.test_size = (736, 1280)

        self.train_ann = 'instances_train.json'
        self.val_ann = 'instances_test.json'
        self.test_ann = 'instances_test.json'
        self.data_dir = os.path.join('datasets', 'soccer')
        self.train_dataset_name = "train"
        self.val_dataset_name = 'test'
        self.test_dataset_name = 'test'
        self.num_classes = 5