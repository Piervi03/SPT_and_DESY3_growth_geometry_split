from __future__ import division
import numpy as np
import set_scaling

def setup(options):
    scaling = set_scaling.SetScaling(options)
    return scaling

def execute(block, scaling):
    if scaling.execute(block):
        block.put_double('likelihoods', 'set_scaling', 0)
        return 0
    else:
        block.put_double('likelihoods', 'set_scaling', -np.inf)
        return 1

def cleanup(config):
    pass
