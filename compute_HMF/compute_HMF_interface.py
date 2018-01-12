from __future__ import division
import numpy as np
from cosmosis.datablock import option_section
import compute_HMF

def setup(options):
    Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
    HMF_calculator = compute_HMF.HMFCalculator(Deltacrit)
    return HMF_calculator

def execute(block, HMF_calculator):
    HMF_calculator.compute_HMF(block)
    return 0

def cleanup(config):
    pass
