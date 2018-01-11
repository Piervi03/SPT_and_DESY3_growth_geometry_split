from __future__ import division
import numpy as np
from cosmosis.datablock import names, option_section
import compute_HMF

cosmo = names.cosmological_parameters

def setup(options):
    Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
    # Initialize HMF calculator
    HMF_calculator = compute_HMF.HMFCalculator(Deltacrit)
    return HMF_calculator

def execute(block, HMF_calculator):
    # print "block.sections", block.sections()
    block.put('hmf', 'dNdlnM', 12.3)

    return 0

def cleanup(config):
    pass
