from __future__ import division
import numpy as np
from cosmosis.datablock import names, option_section
import SPTnumbercount

cosmo = names.cosmological_parameters
likes = names.likelihoods

def setup(options):
    # Initialize number count
    number_count = SPTnumbercount.NumberCount(options)
    return number_count

def execute(block, number_count):
    block[likes, 'ABUNDANCE_LIKE'] = number_count.lnlike(block)

    return 0

def cleanup(config):
    pass
