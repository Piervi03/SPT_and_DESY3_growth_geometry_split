from __future__ import division
import numpy as np
import xarray as xr
import os

from cosmosis.datablock import option_section

import compute_HMF_Tinker08 as compute_HMF


class EmptyClass:
    pass

def setup(options):
    # Print repository status
    path_to_repo = os.path.dirname(__file__)
    os.system("git --git-dir=%s/.git --work-tree=%s status"%(path_to_repo, path_to_repo))
    os.system("git --git-dir=%s/.git --work-tree=%s "%(path_to_repo, path_to_repo)+"show -s --format=%h")
    # Proceed with actual setup
    recalc_HMF = options.get_bool(option_section, 'recalc_HMF', default=True)
    save_HMF_to_disk = options.get_bool(option_section, 'save_HMF_to_disk', default=False)
    Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
    
    if recalc_HMF:
        HMF_calculator = compute_HMF.HMFCalculator(Deltacrit)
    else:
        HMF_calculator = EmptyClass()
        HMF_calculator.HMF = xr.open_dataset('HMF.nc')
    HMF_calculator.recalc_HMF = recalc_HMF
    HMF_calculator.save_HMF_to_disk = save_HMF_to_disk
    return HMF_calculator

def execute(block, HMF_calculator):
    if HMF_calculator.recalc_HMF:
        # Only need cosmo for E(z)-type stuff
        cosmology = {
            'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_nu': block.get_double('cosmological_parameters', 'Omega_nu'),
            'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa')}
        # Matter power spectrum
        z_arr = block.get_double_array_1d('cdm_baryon_power_lin', 'z')
        k_arr = block.get_double_array_1d('cdm_baryon_power_lin', 'k_h')
        Pk = block.get_double_array_nd('cdm_baryon_power_lin', 'p_k')
        # Compute the HMF
        M_arr, dNdlnM_noVol, dNdlnM = HMF_calculator.compute_HMF(cosmology, z_arr, k_arr, Pk)
        # Put it into block
        block.put_double_array_1d('HMF', 'M_arr', M_arr)
        block.put_double_array_1d('HMF', 'z_arr', z_arr)
        block.put_double_array_nd('HMF', 'dNdlnM_unitVol', dNdlnM_noVol)
        block.put_double_array_nd('HMF', 'dNdlnM', dNdlnM)
        if HMF_calculator.save_HMF_to_disk:
            HMF = xr.DataArray(block.get_double_array_nd('HMF', 'dNdlnM'),
                               dims=['z', 'm'],
                               coords={'z': block.get_double_array_1d('HMF', 'z_arr'),
                                       'm': block.get_double_array_1d('HMF', 'M_arr')})
            HMF.to_netcdf('HMF.nc')

    else:
        block.put_double_array_1d('HMF', 'M_arr', np.array(HMF_calculator.HMF['m']))
        block.put_double_array_1d('HMF', 'z_arr', np.array(HMF_calculator.HMF['z']))
        dNdlnM_ = np.array(HMF_calculator.HMF.to_array()[0])
        block.put_double_array_nd('HMF', 'dNdlnM', dNdlnM_)
    return 0

def cleanup(config):
    pass
