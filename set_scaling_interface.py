from cosmosis.datablock import option_section

import set_scaling


def setup(options):
    todo = {}
    for opt in ['doWL', 'doYx', 'doMgas', 'doveldisp', 'dorichness']:
        todo[opt[2:]] = options.get_bool(option_section, opt, False)
    Megacamcalib = {}
    for key in ['MegacamSim', 'Megacam_LSS']:
        Megacamcalib[key] = options.get_double_array_1d(option_section, key)
    for key in ['MegacamMcErr', 'MegacamCenterErr', 'MegacamShearErr', 'MegacamzDistErr', 'MegacamContamCorr']:
        Megacamcalib[key] = options.get_double(option_section, key)
    HSTcalibfile = options.get_string(option_section, 'HSTcalibfile')
    scaling_setter = set_scaling.SetScaling(Megacamcalib, HSTcalibfile)
    return scaling_setter, todo


def execute(block, stuff):
    scaling_setter, todo = stuff
    # Read scaling relation parameters from block
    scaling = {'Dsz': block.get_double('mor_parameters', 'Dsz')}
    if todo['WL']:
        for p in ['MegacamBias', 'HSTbias', 'WLscatter', 'rhoSZWL']:
            scaling[p] = block.get_double('mor_parameters', p)
        if todo['richness']:
            scaling['rhoWLrichness'] = block.get_double('mor_parameters', 'rhoWLrichness')
    if todo['richness']:
        for p in ['Drichness', 'rhoSZrichness']:
            scaling[p] = block.get_double('mor_parameters', p)
    if todo['Yx'] or todo['Mgas']:
        for p in ['Dx', 'rhoSZX']:
            scaling[p] = block.get_double('mor_parameters', p)
        if todo['richness']:
            scaling['rhoXrichness'] = block.get_double('mor_parameters', 'rhoXrichness')
        if todo['WL']:
            scaling['rhoWLX'] = block.get_double('mor_parameters', 'rhoWLX')
    # See if DES model is defined, else skip
    if block.has_value('mor_parameters', 'DESwl_z'):
        for p in ['DES_b_dev_0', 'DES_b_dev_1', 'DES_b_dev_2', 'DES_b_dev_m', 'DESwl_scatter_m_mean',
                  'DES_s_dev_0', 'DES_s_dev_1', 'DES_s_dev_2', 'DES_s_dev_m', 'DESwl_scatter_m_std',
                  'DES_m_piv']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in ['DESwl_z', 'DESwl_scatter_mean', 'DESwl_scatter_std']:
            scaling[p] = block.get_double_array_1d('mor_parameters', p)
    # Set everything
    if scaling_setter.execute(todo, scaling):
        # Add lensing stuff to block
        if todo['WL']:
            for p in ['bWL_Megacam', 'DWL_Megacam']:
                block.put_double('mor_parameters', p, scaling[p])
            for name in scaling_setter.HSTcalib['SPT_ID']:
                block.put_double('mor_parameters', 'bWL_HST_%s' % name, scaling['bWL_HST'][name])
                block.put_double('mor_parameters', 'DWL_HST_%s' % name, scaling['DWL_HST'][name])
        return 0
    else:
        print("set scaling", flush=True)
        return 1


def cleanup(config):
    pass
