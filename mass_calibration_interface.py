from __future__ import division
import numpy as np
import mass_calibration

measure_time = False
if measure_time:
    import time

def setup(options):
    masscalibration = mass_calibration.MassCalibration(options)
    return masscalibration

def execute(block, masscalibration):
    if measure_time:
        t0 = time.time()
    lnlike = masscalibration.lnlike(block)
    if measure_time:
        print "Mass calibration took", time.time()-t0
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
