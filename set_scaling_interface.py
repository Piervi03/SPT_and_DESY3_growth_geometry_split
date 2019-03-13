from __future__ import division
import set_scaling

def setup(options):
    scaling = set_scaling.SetScaling(options)
    return scaling

def execute(block, scaling):
    if scaling.execute(block):
        return 0
    else:
        return 1

def cleanup(config):
    pass
