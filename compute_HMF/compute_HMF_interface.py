from __future__ import division
import numpy as np

from cosmosis.datablock import names, option_section

cosmo = names.cosmological_parameters

def setup(options):
    return 0

def execute(block, config):
    print block.sections()

def cleanup(config):
    pass
