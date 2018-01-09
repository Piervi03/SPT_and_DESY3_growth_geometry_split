from __future__ import division
import numpy as np

from cosmosis.datablock import names, option_section

cosmo = names.cosmological_parameters

def setup(options):
    return 0

def execute(block, config):
    print block.sections()
    print block['matter_power_lin', '_cosmosis_order_p_k']
    print block['matter_power_lin', 'p_k'].shape

def cleanup(config):
    pass

