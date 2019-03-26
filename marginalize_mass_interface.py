from __future__ import division
import numpy as np
import marginalize_mass

def setup(options):
    marge_mass = marginalize_mass.MarginalizeMass(options)
    return marge_mass

def execute(block, marge_mass):
    marge_mass.do_it(block)

    return 0

def cleanup(config):
    pass
