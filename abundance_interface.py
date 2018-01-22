from __future__ import division
import numpy as np
import abundance

measure_time = False
if measure_time:
    import time

def setup(options):
    number_count = abundance.NumberCount(options)
    return number_count

def execute(block, number_count):
    if measure_time:
        t0 = time.time()
    lnlike = float(number_count.lnlike(block))
    if measure_time:
        print "Abundance took", time.time()-t0
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
