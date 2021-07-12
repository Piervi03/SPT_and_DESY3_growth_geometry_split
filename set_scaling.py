from __future__ import division
import numpy as np
from math import sqrt as msqrt
import imp

import scaling_relations

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

        # HST
        zDistShearErr = msqrt(self.WLcalib['HSTzDistErr']**2 + self.WLcalib['HSTshearErr']**2)
        scaling['bWL_HST'], scaling['DWL_HST'] = {}, {}
        for name in self.WLcalib['HSTsim'].keys():
            # bias = bSim + bMassModel + (bN(z)+bShearCal)
            mass_model_err = msqrt(self.WLcalib['HSTsim'][name]['bias'][1]**2 + self.WLcalib['HSTmcErr']**2 + self.WLcalib['HSTsim'][name]['center_err']**2)
            scaling['bWL_HST'][name] = self.WLcalib['HSTsim'][name]['bias'][0] \
                + scaling['WLbias'] * mass_model_err \
                + scaling['HSTbias'] * zDistShearErr
            # lognormal scatter
            DWL_HST = self.WLcalib['HSTsim'][name]['bias'][2] + scaling['WLscatter']*self.WLcalib['HSTsim'][name]['bias'][3]
            scaling['DWL_HST'][name] = DWL_HST
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
            # SZ WL richness covariance matrix
            cov = [[DWL_HST**2, scaling['rhoWLrichness']*DWL_HST*scaling['Drichness'], scaling['rhoSZWL']*scaling['Dsz']*DWL_HST],
                   [scaling['rhoWLrichness']*DWL_HST*scaling['Drichness'], scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness']],
                   [scaling['rhoSZWL']*scaling['Dsz']*DWL_HST, scaling['rhoSZrichness']*scaling['Dsz']*scaling['Drichness'], scaling['Dsz']**2]]
            if np.linalg.det(cov)<THRESHOLD:
                return False
            scaling['cov_HST_richness_SZ_%s'%name] = np.array(cov)
 

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
        z = np.array([.25, .25, 1., 1.])
        M = np.array([1e13, 1e16, 1e13, 1e16])
        DES_scatter = scaling_relations.WLscatter('main', M, z, scaling)
        dets = [np.linalg.det([[DES_scatter[i]**2, scaling['rhoSZWL']*scaling['Dsz']*DES_scatter[i]],
                               [scaling['rhoSZWL']*scaling['Dsz']*DES_scatter[i], scaling['Dsz']**2]])
                for i in range(4)]
        if np.any(np.array(dets)<THRESHOLD):
            return False


        # X-ray and WL: Megacam
        cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
            [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
            [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        scaling['cov_Megacam_X_SZ'] = np.array(cov)

        # DES WL and Richness [DES WL, richness, SZ]
        z = np.array([.25, .25, 1., 1.])
        M = np.array([1e13, 1e16, 1e13, 1e16])
        DES_scatter = scaling_relations.WLscatter('main', M, z, scaling)
        dets = [np.linalg.det([[DES_scatter[i]**2, scaling['rhoWLrichness']*DES_scatter[i]*scaling['Drichness'], scaling['rhoSZWL']*DES_scatter[i]*scaling['Dsz']],
                               [scaling['rhoWLrichness']*DES_scatter[i]*scaling['Drichness'], scaling['Drichness']**2, scaling['rhoSZrichness']*scaling['Drichness']*scaling['Dsz']],
                               [scaling['rhoSZWL']*DES_scatter[i]*scaling['Dsz'], scaling['rhoSZrichness']*scaling['Drichness']*scaling['Dsz'], scaling['Dsz']**2]])
                for i in range(4)]
        if np.any(np.array(dets)<THRESHOLD):
            return False


        return True
