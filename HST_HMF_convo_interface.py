from __future__ import division
import HST_HMF_convo

def setup(options):
    multi_obs_convolution = HST_HMF_convo.MultiObsConvolution(options)
    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    multi_obs_convolution.execute(block)
    return 0

def cleanup(config):
    pass
