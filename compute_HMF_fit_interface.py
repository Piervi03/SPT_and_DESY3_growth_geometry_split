import numpy as np
import xarray as xr
import os

from cosmosis.datablock import option_section

import compute_HMF_Bocquet16
import compute_HMF_Tinker08
import compute_HMF_Tinker10


class EmptyClass:
    pass


def setup(options):
    # Print repository status
    path_to_repo = os.path.dirname(__file__)
    os.system("git --git-dir=%s/.git --work-tree=%s status" % (path_to_repo, path_to_repo))
    os.system("git --git-dir=%s/.git --work-tree=%s " % (path_to_repo, path_to_repo)+"show -s --format=%h")
    # Proceed with actual setup
    tmp = options.get_double_array_1d(option_section, 'z_arr')
    z_arr = np.linspace(tmp[0], tmp[1], int(tmp[2]))
    tmp = options.get_double_array_1d(option_section, 'M_arr')
    M_arr = np.logspace(tmp[0], tmp[1], int(tmp[2]))
    fitting_function = options.get_string(option_section, 'fitting_function')
    recalc_HMF = options.get_bool(option_section, 'recalc_HMF', default=True)
    save_HMF_to_disk = options.get_bool(option_section, 'save_HMF_to_disk', default=False)
    Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
    if recalc_HMF:
        if fitting_function == 'Tinker08':
            HMF_calculator = compute_HMF_Tinker08.HMFCalculator(Deltacrit, z_arr, M_arr)
        elif fitting_function == 'Tinker10':
            HMF_calculator = compute_HMF_Tinker10.HMFCalculator(Deltacrit, z_arr, M_arr)
        elif fitting_function == 'Bocquet16':
            HMF_calculator = compute_HMF_Bocquet16.HMFCalculator(Deltacrit, z_arr, M_arr)
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
            'h': block.get_double('cosmological_parameters', 'hubble')/100.,
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa'),
            'HMFbias': block.get_double('cosmological_parameters', 'HMFbias', default=1.),
            'HMFslope': block.get_double('cosmological_parameters', 'HMFslope', default=0.)
        }
        # cdm+bar power spectrum (w/o neutrinos)
        z, k, Pk = block.get_grid('cdm_baryon_power_lin', 'z', 'k_h', 'p_k')
        # Compute the HMF
        dNdlnM_noVol, dNdlnM = HMF_calculator.compute_HMF(cosmology, z, k, Pk)
        dNdlnM_noVol *= cosmology['HMFbias'] + cosmology['HMFslope']*np.log(HMF_calculator.M_arr/1e14)
        dNdlnM *= cosmology['HMFbias'] + cosmology['HMFslope']*np.log(HMF_calculator.M_arr/1e14)
        # Put it into block
        block.put_grid('HMF', 'z_arr', HMF_calculator.z_arr, 'M_arr', HMF_calculator.M_arr, 'dNdlnM', dNdlnM)
        block.put_double_array_nd('HMF', 'dNdlnM_unitVol', dNdlnM_noVol)
        if HMF_calculator.save_HMF_to_disk:
            HMF = xr.DataArray(block.get_double_array_nd('HMF', 'dNdlnM'),
                               dims=['z', 'm'],
                               coords={'z': block.get_double_array_1d('HMF', 'z_arr'),
                                       'm': block.get_double_array_1d('HMF', 'M_arr')})
            HMF.to_netcdf('HMF.nc')

    else:
        block.put_grid('HMF',
                       'z_arr', np.array(HMF_calculator.HMF['z']),
                       'M_arr', np.array(HMF_calculator.HMF['m']),
                       'dNdlnM', np.array(HMF_calculator.HMF.to_array()[0]))
    return 0


def cleanup(config):
    pass
