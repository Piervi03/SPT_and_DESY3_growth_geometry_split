import numpy as np
from scipy.interpolate import RectBivariateSpline

import MiraTitanHMFemulator
import cosmo
import Mconversion_concentration


class HMFCalculator:
    def __init__(self, Deltacrit, mcType, z_arr, M_arr):
        self.HMFemu = MiraTitanHMFemulator.Emulator()
        self.Deltacrit = Deltacrit
        self.mcType = mcType
        self.z_arr = z_arr
        self.M_arr = M_arr

    def compute_HMF(self, cosmology):
        if not self.HMFemu.validate_params(cosmology):
            return 1

        if self.Deltacrit == 200.:
            self.dNdlnM_unitVol = self.HMFemu.predict(cosmology, self.z_arr, self.M_arr, get_errors=False)[0]

        else:
            emu_dict = self.HMFemu.predict_raw_emu(cosmology)
            HMF_interp_input = np.full((len(self.HMFemu.z_arr_asc), 3501), -np.inf)
            for i, emu_z in enumerate(self.HMFemu.z_arr_asc):
                HMF_interp_input[i, :len(emu_dict[emu_z]['HMF'])] = np.log(emu_dict[emu_z]['HMF'])
            HMF_interp = RectBivariateSpline(self.HMFemu.z_arr_asc, np.linspace(13, 16.5, 3501), HMF_interp_input, kx=1, ky=1)

            MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, cosmology,
                                                                      setup_interp=True, interp_massdef=self.Deltacrit)
            lnM200 = MCrel.lnM_to_lnM200(self.z_arr, np.log(self.M_arr))

            self.dNdlnM_unitVol = np.empty((len(self.z_arr), len(self.M_arr)))
            for z_id, z in enumerate(self.z_arr):
                dlnM200_dlnMDelta = (lnM200[z_id, 2:]-lnM200[z_id, :-2]) / (np.log(self.M_arr[2:]/self.M_arr[:-2]))
                dlnM200_dlnMDelta = np.insert(dlnM200_dlnMDelta, 0, (lnM200[z_id, 1]-lnM200[z_id, 0])/np.log(self.M_arr[1]/self.M_arr[0]))
                dlnM200_dlnMDelta = np.append(dlnM200_dlnMDelta, (lnM200[z_id, -1]-lnM200[z_id, -2])/np.log(self.M_arr[-1]/self.M_arr[-2]))
                self.dNdlnM_unitVol[z_id] = np.exp(HMF_interp(z, lnM200[z_id]/np.log(10))) * dlnM200_dlnMDelta

        # Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        self.dNdlnM = self.dNdlnM_unitVol * deltaV[:, None]

        return 0
