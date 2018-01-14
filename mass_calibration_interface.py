from __future__ import division
import numpy as np
import masscalib

def setup(options):
    mass_calibration = masscalib.MassCalibration(options)
    return mass_calibration

def execute(block, mass_calibration):
    lnlike = mass_calibration.lnlike(block)
    block.put_double('likelihoods', 'MASS_CALIBRATION_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
