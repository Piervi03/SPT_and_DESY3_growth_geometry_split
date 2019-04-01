from __future__ import division
import numpy as np
from cosmosis.datablock import option_section
import marginalize_mass

def setup(options):
    ##### Global variables
    SZmPivot = options.get_double(option_section, 'SZmPivot')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')

    marge_mass = marginalize_mass.MarginalizeMass(SPTcatalogfile, SPT_survey_fields,
                                                  SZmPivot)

    return marge_mass

def execute(block, marge_mass):
    # Only need cosmo for E(z)-type stuff
    marge_mass.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
        'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    marge_mass.Asz = block.get_double('mor_parameters', 'Asz')
    marge_mass.Bsz = block.get_double('mor_parameters', 'Bsz')
    marge_mass.Csz = block.get_double('mor_parameters', 'Csz')
    marge_mass.Dsz = block.get_double('mor_parameters', 'Dsz')
    # Halo mass function
    marge_mass.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    marge_mass.HMF['len_z'] = len(marge_mass.HMF['z_arr'])

    #### Get marginalized mass draws
    mass_arr = marge_mass.do_it()

    ##### Put back into block
    for i in range(len(mass_arr)):
        block.put_double('marge_mass', 'M500_%d'%i, mass_arr[i,0])
        block.put_double('marge_mass', 'M200_%d'%i, mass_arr[i,1])
        block.put_double('marge_mass', 'weight_%d'%i, mass_arr[i,2])

    return 0

def cleanup(config):
    pass
