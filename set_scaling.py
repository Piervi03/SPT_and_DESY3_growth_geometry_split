from __future__ import division
import numpy as np

from cosmosis.datablock import option_section

THRESHOLD = 1e-8

class SetScaling:

    def __init__(self, options):
        # WL simulation calibration data
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration


    def execute(self, block):
        """Set total (or effective) bias and scatter for Megacam and DES using
        the simulation calibration numbers and the nuissance parameters. Set
        possible covariance matrices between all observables we're currently
        analyzing. The scatter in velocity dispersions depends on cluster
        properties and therefore cannot be pre-computed. Return: (bool) whether
        or not all covariance matrices can be inverted (by checking whether all
        determinants are >= THRESHOLD) """
        scaling = {}
        for p in ['Dsz', 'Dx', 'Drichness', 'WLbias', 'WLscatter']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in ['MegacamBias', 'MegacamScatterLSS', 'DESbias', 'DESscatterLSS']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in ['rhoSZWL', 'rhoSZX', 'rhoWLX', 'rhoSZrichness', 'rhoXdisp', 'rhoSZdisp']:
            scaling[p] = block.get_double('mor_parameters', p)


        # Megacam
        massModelErr = (self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2)**.5
        zDistShearErr = (self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2 + self.WLcalib['MegacamContamCorr']**2)**.5
        # bias = bSim + bMassModel + (bN(z)+bShearCal)
        scaling['bWL_Megacam'] = self.WLcalib['MegacamSim'][0] + scaling['WLbias']*massModelErr + scaling['MegacamBias']*zDistShearErr
        # lognormal scatter
        scaling['DWL_Megacam'] = self.WLcalib['MegacamSim'][2] + scaling['WLscatter']*self.WLcalib['MegacamSim'][3]

        # DES
        massModelErr = (self.WLcalib['DESsim'][1]**2 + self.WLcalib['DESmcErr']**2 + self.WLcalib['DEScenterErr']**2)**.5
        zDistShearErr = (self.WLcalib['DESzDistErr']**2 + self.WLcalib['DESshearErr']**2 + self.WLcalib['DEScontamCorr']**2)**.5
        # bias = bSim + bFitParam * err(bSim)
        scaling['bWL_DES'] = self.WLcalib['DESsim'][0] + scaling['WLbias']*massModelErr + scaling['DESbias']*zDistShearErr
        # D^2 = Dint^2 + (DSim + DErrParam * err(DSim))^2
        scaling['DWL_DES'] = self.WLcalib['DESsim'][2] + scaling['WLscatter']*self.WLcalib['DESsim'][3]



        ##### one follow-up observable
        # X-ray
        cov = [[scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
        [scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        block.put_double_array_nd('scaling', 'cov_X_SZ', np.array(cov))

        # Richness
        cov = [[scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
            [scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        block.put_double_array_nd('scaling', 'cov_rich_SZ', np.array(cov))

        # WL: Megacam
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        block.put_double_array_nd('scaling', 'cov_Megacam_SZ', np.array(cov))

        # WL: DES
        cov = [[scaling['DWL_DES']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
                [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['Dsz']**2]]
            if np.linalg.det(cov) < THRESHOLD:
                return False
        block.put_double_array_nd('scaling', 'cov_DES_SZ', np.array(cov))


        ##### two follow-up observables

        # X-ray and WL: Megacam
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        block.put_double_array_nd('scaling', 'cov_Megacam_X_SZ', np.array(cov))

        # X-ray and WL: DES
        cov = [[scaling['DWL_DES']**2, scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
            [scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        block.put_double_array_nd('scaling', 'cov_DES_X_SZ', np.array(cov))

        return True
