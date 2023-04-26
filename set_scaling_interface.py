from __future__ import division

from cosmosis.datablock import option_section

import set_scaling


def setup(options):
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
    HSTcalibfile = options.get_string(option_section, 'HSTcalibfile')
    scaling_setter = set_scaling.SetScaling(WLsimcalibfile, HSTcalibfile)
    return scaling_setter


def execute(block, scaling_setter):
    # Read scaling relation parameters from block
    scaling = {}
    for p in ['Dsz', 'Dx', 'Drichness',
              'MegacamBias', 'HSTbias', 'WLscatter',
              'rhoSZWL', 'rhoSZX', 'rhoWLX', 'rhoSZrichness', 'rhoXdisp', 'rhoSZdisp', 'rhoWLrichness']:
        scaling[p] = block.get_double('mor_parameters', p)
    # See if DES model is defined, else skip
    if block.has_value('mor_parameters', 'DESwl_z'):
        for p in ['DES_b_dev_0', 'DES_b_dev_1', 'DES_b_dev_2', 'DES_b_dev_m', 'DESwl_scatter_m_mean',
                  'DES_s_dev_0', 'DES_s_dev_1', 'DES_s_dev_2', 'DES_s_dev_m', 'DESwl_scatter_m_std',
                  'DES_m_piv']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in ['DESwl_z', 'DESwl_scatter_mean', 'DESwl_scatter_std']:
            scaling[p] = block.get_double_array_1d('mor_parameters', p)
    # Set everything
    if scaling_setter.execute(scaling):
        # Put into block
        for p in ['bWL_Megacam', 'DWL_Megacam']:
            block.put_double('mor_parameters', p, scaling[p])
        for name in scaling_setter.HSTcalib['SPT_ID']:
            block.put_double('mor_parameters', 'bWL_HST_%s'%name, scaling['bWL_HST'][name])
            block.put_double('mor_parameters', 'DWL_HST_%s'%name, scaling['DWL_HST'][name])
        block.put_double('likelihoods', 'set_scaling_like', 0)
        return 0
    else:
        print("set scaling", flush=True)
        return 1


def cleanup(config):
    pass
