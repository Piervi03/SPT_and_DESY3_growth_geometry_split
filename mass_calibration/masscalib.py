from __future__ import division
import numpy as np
from multiprocessing import Pool
import cosmo, Mconversion_concentration, lensing
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import interpolate
from scipy import signal
import scipy.ndimage
import scipy.special as ss
import os
from astropy.table import Table
import imp

# Cosmosis stuff
from cosmosis.datablock import option_section

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1}
getpull = False

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg, **kwarg):
    return MassCalibration.clusterlike(*arg, **kwarg)

################################################################################
class MassCalibration:

    def __init__(self, options):
        ##### Config parameters
        self.doWL = options.get_bool(option_section, 'doWL')
        self.doXray = options.get_bool(option_section, 'doXray')
        self.dorichness = options.get_bool(option_section, 'dorichness')
        self.doveldisp = options.get_bool(option_section, 'doveldisp')
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.XraymPivot = options.get_double(option_section, 'XraymPivot')
        self.richmPivot = options.get_double(option_section, 'richmPivot')
        self.YXPARAM = options.get_string(option_section, 'YXPARAM')
        self.mcType = options.get_string(option_section, 'mcType')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        self.NPROC = options.get_int(option_section, 'NPROC')
        self.XrayProfileHandling = options.get_string(option_section, 'XrayProfileHandling')
        assert self.XrayProfileHandling in ('fixed', 'old', 'modelMgasPL'), "invalid XrayProfileHandling"
        ##### SPT survey
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        SPTcatalogfile = SPTdata.SPTcatalogfile
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        self.SPTfieldNames = SPTdata.SPTfieldNames
        self.SPTfieldCorrection = SPTdata.SPTfieldCorrection
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.XraySample = SPTdata.XraySample
        ##### WL simulation calibration
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        # Weak lensing
        if self.doWL:
            self.WL = lensing.SPTlensing(options, self.catalog)


    # Return ln(likelihood) for the whole sample
    def lnlike(self, block):

        ##### Evaluate the individual likelihoods
        len_data = 10

        if self.NPROC==0:
            # Iterate through cluster list
            likelihoods = np.array([self.clusterlike(i) for i in range(len_data)])
        else:
            # Launch a multiprocessing pool and get the likelihoods
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len_data, range(len_data))
            likelihoods = pool.map(unwrap_self_f, argin)
            pool.close()
        lnlike = np.sum(np.log(likelihoods))

        return lnlike



################################################################################
# Likelihood of given cluster
    def clusterlike(self, i):
        return float(i+1)
