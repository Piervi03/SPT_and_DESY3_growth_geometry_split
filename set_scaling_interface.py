from __future__ import division

from cosmosis.datablock import option_section

import set_scaling

def setup(options):
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
    scaling_setter = set_scaling.SetScaling(WLsimcalibfile)
    return scaling_setter

def execute(block, scaling_setter):
    # Read scaling relation parameters from block
    scaling = {}
    for p in ['Dsz', 'Dx', 'Drichness', 'WLbias', 'WLscatter']:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in ['MegacamBias', 'HSTbias']:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in ['rhoSZWL', 'rhoSZX', 'rhoWLX', 'rhoSZrichness', 'rhoXdisp', 'rhoSZdisp', 'rhoWLrichness']:
        scaling[p] = block.get_double('mor_parameters', p)
    # Set everything
    if scaling_setter.execute(scaling):
        # Put into block
        for p in ['bWL_Megacam',]:
            block.put_double('mor_parameters', p, scaling[p])
        for p in ['cov_X_SZ', 'cov_richness_SZ', 'cov_Megacam_SZ', 'cov_Megacam_X_SZ']:
            block.put_double_array_nd('mor_parameters', p, scaling[p])
        for name in scaling_setter.WLcalib['HSTsim'].keys():
            block.put_double('mor_parameters', 'bWL_HST_%s'%name, scaling['bWL_HST'][name])
            block.put_double_array_nd('mor_parameters', 'cov_HST_SZ_%s'%name, scaling['cov_HST_SZ_%s'%name])
            block.put_double_array_nd('mor_parameters', 'cov_HST_X_SZ_%s'%name, scaling['cov_HST_X_SZ_%s'%name])
        return 0
    else:
        return 1

def cleanup(config):
    pass
