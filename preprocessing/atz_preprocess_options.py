""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch


# pylint: disable=C0103,C0301,R0903,W0622

class ATZPreprocessOptions:
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        default_save = "/Users/soumen/Desktop/Skip-Attention-GAN/customdataset/atz"
        self.parser.add_argument('--ablation', default=0,
                                 help='Number of "abnormal" images required in the dataset. normal=abnormal*2')
        self.parser.add_argument('--save_path', default=default_save, help='path to prepare and save the data')
        self.parser.add_argument('--save_image', action='store_true', default=False,
                                 help='Create a small dataset for ablation study')
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        return self.opt