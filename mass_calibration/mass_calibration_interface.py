from __future__ import division
import numpy as np
from cosmosis.datablock import names, option_section
import masscalib

cosmo = names.cosmological_parameters
likes = names.likelihoods

def setup(options):
    # Initialize number count
    mass_calibration = masscalib.MassCalibration(options)
    return mass_calibration

def execute(block, mass_calibration):
    block[likes, 'MASS_CALIBRATION_LIKE'] = mass_calibration.lnlike(block)

    return 0

def cleanup(config):
    pass
