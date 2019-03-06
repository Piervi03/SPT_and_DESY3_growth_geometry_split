from __future__ import division
import HMF_convo

def setup(options):
    multi_obs_convolution = HMF_convo.MultiObsConvolution(options)
    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    multi_obs_convolution.execute(block)
    return 0

def cleanup(config):
    pass
