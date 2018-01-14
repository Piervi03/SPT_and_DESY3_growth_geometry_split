from __future__ import division
import numpy as np
import SPTnumbercount

def setup(options):
    number_count = SPTnumbercount.NumberCount(options)
    return number_count

def execute(block, number_count):
    lnlike = float(number_count.lnlike(block))
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
