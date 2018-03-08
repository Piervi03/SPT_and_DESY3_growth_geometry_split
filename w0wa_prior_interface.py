from __future__ import division
import numpy as np
 
def setup(options):
    pass

def execute(block, HMF_calculator):
    w0 = block.get_double('cosmological_parameters', 'w')
    wa = block.get_double('cosmological_parameters', 'wa')
    lnlike = 0 if (w0+wa)<0 else -np.inf
    block.put_double('likelihoods', 'W0WA_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
