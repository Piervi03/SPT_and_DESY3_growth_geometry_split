from __future__ import division
import numpy as np
import mass_calibration

if __debug__:
    import time

def setup(options):
    masscalibration = mass_calibration.MassCalibration(options)
    return masscalibration

def execute(block, masscalibration):
    if __debug__:
        t0 = time.time()
    lnlike = masscalibration.lnlike(block)
    if __debug__:
        print "Mass calibration took", time.time()-t0
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
