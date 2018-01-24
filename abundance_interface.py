from __future__ import division
import numpy as np
import abundance

if __debug__:
    import time

def setup(options):
    number_count = abundance.NumberCount(options)
    return number_count

def execute(block, number_count):
    if __debug__:
        t0 = time.time()
    lnlike = float(number_count.lnlike(block))
    if __debug__:
        print "Abundance took", time.time()-t0
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
