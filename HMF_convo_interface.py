from __future__ import division
import HMF_convo

def setup(options):
    convolution_calc = HMF_convo.Convolution(options)
    return convolution_calc

def execute(block, convolution_calc):
    convolution_calc.compute(block)
    return 0

def cleanup(config):
    pass
