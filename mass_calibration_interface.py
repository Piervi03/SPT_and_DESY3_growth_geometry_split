from __future__ import division
import numpy as np
import mass_calibration

def setup(options):
    masscalibration = mass_calibration.MassCalibration(options)
    return masscalibration

def execute(block, masscalibration):
    lnlike = masscalibration.lnlike(block)
    if np.isneginf(lnlike):
        return 1
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
