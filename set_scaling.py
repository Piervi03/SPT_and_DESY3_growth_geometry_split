from __future__ import division
import numpy as np
from math import sqrt as msqrt
import imp

THRESHOLD = 1e-8

class SetScaling:

    def __init__(self, WLsimcalibfile):
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

    def execute(self, scaling):
        """Set total (or effective) bias and scatter for Megacam and DES using
        the simulation calibration numbers and the nuissance parameters. Set
        possible covariance matrices between all observables we're currently
        analyzing. The scatter in velocity dispersions depends on cluster
        properties and therefore cannot be pre-computed. Return: (bool) whether
        or not all covariance matrices can be inverted (by checking whether all
        determinants are >= THRESHOLD) """

        # Megacam
        massModelErr = msqrt(self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2)
        zDistShearErr = msqrt(self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2 + self.WLcalib['MegacamContamCorr']**2)
        # bias = bSim + bMassModel + (bN(z)+bShearCal)
        scaling['bWL_Megacam'] = self.WLcalib['MegacamSim'][0] + scaling['WLbias']*massModelErr + scaling['MegacamBias']*zDistShearErr
        # lognormal scatter
        scaling['DWL_Megacam'] = self.WLcalib['MegacamSim'][2] + scaling['WLscatter']*self.WLcalib['MegacamSim'][3]

        # DES
        massModelErr = msqrt(self.WLcalib['DESsim'][1]**2 + self.WLcalib['DESmcErr']**2 + self.WLcalib['DEScenterErr']**2)
        # bias = bSim + bFitParam * err(bSim)
        scaling['bWL_DES'] = self.WLcalib['DESsim'][0] + scaling['WLbias']*massModelErr
        # lognormal scatter
        scaling['DWL_DES'] = self.WLcalib['DESsim'][2] + scaling['WLscatter']*self.WLcalib['DESsim'][3]

        # HST
        zDistShearErr = msqrt(self.WLcalib['HSTzDistErr']**2 + self.WLcalib['HSTshearErr']**2)
        scaling['bWL_HST'] = {}
        for name in self.WLcalib['HSTsim'].keys():
            # bias = bSim + bMassModel + (bN(z)+bShearCal)
            mass_model_err = msqrt(self.WLcalib['HSTsim'][name]['bias'][1]**2 + self.WLcalib['HSTmcErr']**2 + self.WLcalib['HSTcenterErr']**2)
            scaling['bWL_HST'][name] = self.WLcalib['HSTsim'][name]['bias'][0] \
                + scaling['WLbias'] * mass_model_err \
                + scaling['HSTbias'] * zDistShearErr
            # lognormal scatter
            DWL_HST = self.WLcalib['HSTsim'][name]['bias'][2] + scaling['WLscatter']*self.WLcalib['HSTsim'][name]['bias'][3]
            # SZ WL covariance matrix
            cov = [[DWL_HST**2, scaling['rhoSZWL']*scaling['Dsz']*DWL_HST],
                   [scaling['rhoSZWL']*scaling['Dsz']*DWL_HST, scaling['Dsz']**2]]
            if np.linalg.det(cov)<THRESHOLD:
                return False
            scaling['cov_HST_SZ_%s'%name] = np.array(cov)
            # SZ WL X covariance matrix
            cov = [[DWL_HST**2, scaling['rhoWLX']*DWL_HST*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*DWL_HST],
                   [scaling['rhoWLX']*DWL_HST*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
                   [scaling['rhoSZWL']*scaling['Dsz']*DWL_HST, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
            if np.linalg.det(cov)<THRESHOLD:
                return False
            scaling['cov_HST_X_SZ_%s'%name] = np.array(cov)



        ##### one follow-up observable
        # X-ray
        cov = [[scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
        [scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_X_SZ'] = np.array(cov)

        # Richness
        cov = [[scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
            [scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_richness_SZ'] = np.array(cov)

        # WL: Megacam
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_Megacam_SZ'] = np.array(cov)

        # WL: DES
        cov = [[scaling['DWL_DES']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
                [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_DES_SZ'] = np.array(cov)


        ##### two follow-up observables

        # X-ray and WL: Megacam
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_Megacam_X_SZ'] = np.array(cov)

        # X-ray and WL: DES
        cov = [[scaling['DWL_DES']**2, scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
            [scaling['rhoWLX']*scaling['DWL_DES']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_DES_X_SZ'] = np.array(cov)

        # Richness and WL: [WL, richness, SZ]
        cov = [[scaling['DWL_DES']**2, scaling['rhoWLrichness']*scaling['DWL_DES']*scaling['Drichness'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES']],
            [scaling['rhoWLrichness']*scaling['DWL_DES']*scaling['Drichness'], scaling['Dx']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_DES'], scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_DES_richness_SZ'] = np.array(cov)

        return True
