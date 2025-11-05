import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table


def test_mockgenerator(tmp_path_factory):
    """Create mock catalog using `mockinput.py` setup file."""
    import mockgenerator
    f_name = tmp_path_factory.mktemp('data') / "mock.fits"
    mockgenerator.main('mockinput.py', f_name)


class TestClass:
    """Default parameters"""
    cosmology = {'Omega_m': .28, 'Omega_l': .72, 'Ombh2': .022, 'Omnuh2': .0006,
                 'h': 0.7,
                 'w0': -1., 'wa': 0.,
                 'n_s': .96, 'ln1e10As': 3.001}
    cosmology['Omega_nu'] = cosmology['Omnuh2']/cosmology['h']**2
    cosmology['Omega_b'] = cosmology['Ombh2']/cosmology['h']**2
    scaling = {'Asz': .96, 'Bsz': 1.5, 'Csz': .5, 'Dsz': .2, 'zeta_min': 1., 'SPECS_calib': 1.,
               'Esz': 0,
               'Delta_Csz_ECS': -.09, 'Delta_Csz_500d': .26,
               'WLbias': 0., 'WLscatter': 0.,
               'HSTbias': 0., 'HSTscatterLSS': 5.6e13,
               'MegacamBias': 0., 'MegacamScatterLSS': 6.3e13,
               'DWL_Megacam': .3, 'bWL_Megacam': 1,
               'DESbias': 0., 'DESscatterLSS': 6.3e13,
               'Adisp': 939., 'Bdisp': 2.91, 'Cdisp': .33, 'Ddisp0': .2, 'DdispN': 3.,
               'Arichness': 70., 'Brichness': 1., 'Crichness': 0., 'Drichness': .2,
               'Ax': 6.5, 'Bx': .57, 'Cx': -.4, 'Dx': .12, 'Ex': 0,
               'slope_MgR': 1.16, 'slope_MgR_std': .016,
               'rhoSZrichness': 0., 'rhoSZdisp': 0., 'rhoSZX': 0., 'rhoSZWL': 0.,
               'rhoWLX': 0., 'rhoWLrichness': 0.,
               'rhoXrichness': 0,
               'SZmPivot': 3e14,
               'XraymPivot': 5e14,
               'richmPivot': 3e14,
               'YXPARAM': 'SPT_XVP',
               }
    # Arrays for mass function
    z_arr_pk = np.linspace(0, 2, 21)
    Deltacrit = 200.
    z_arr = np.linspace(0, 2, 201)
    M_arr = np.logspace(13, 16, 301)
    lnM_arr = np.log(M_arr)
    # Observables for multi-obs convolutions
    observable_pairs = ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep', 'SZ']
    pairs_zmin = [.25, .25, .25]
    pairs_zmax = [1.78, 1.78, 1.78]
    pairs_Nz = [154, 154, 154]
    # MCMF lambda
    tmp = np.loadtxt('data/MCMF_lambda_min.txt', unpack=True)
    surveyCutLambda = {'shallow': interp1d(tmp[0], tmp[1], kind='linear'),
                       'deep': interp1d(tmp[0], tmp[2], kind='linear')}
    richness_scatter_model = 'lognormal'

    def test_scaling(self):
        """Check and set scaling relation parameters and covariance matrices."""
        import set_scaling
        scaling_setter = set_scaling.SetScaling('data/WLsimcalib_data_Megacam.py', 'data/HST-39_200_X.txt')
        scaling_setter.execute(self.scaling)
        return self.scaling

    def test_pk_baccoemu(self):
        """Compute matter power spectrum"""
        import baccoemu
        emulator = baccoemu.Matter_powerspectrum()
        k, Pk = emulator.get_linear_pk(omega_matter=self.cosmology['Omega_m'],
                                       omega_baryon=self.cosmology['Omega_b'],
                                       hubble=self.cosmology['h'],
                                       ns=self.cosmology['n_s'],
                                       w0=self.cosmology['w0'],
                                       wa=self.cosmology['wa'],
                                       neutrino_mass=self.cosmology['Omnuh2']*94.,
                                       A_s=1e-10*np.exp(self.cosmology['ln1e10As']),
                                       expfactor=1./(1.+self.z_arr_pk),
                                       cold=True)
        return k, Pk

    def test_compute_HMF_Tinker08(self):
        """Compute halo mass function"""
        import compute_HMF_Tinker08
        HMF_calculator = compute_HMF_Tinker08.HMFCalculator(self.Deltacrit, self.z_arr, self.M_arr)
        k, Pk = self.test_pk_baccoemu()
        dNdlnM_noVol, dNdlnM = HMF_calculator.compute_HMF(self.cosmology, self.z_arr_pk, k, Pk)
        return dNdlnM

    def test_HMF_convo(self):
        """Compute multi-observable mass functions with correlated intrinsic
        scatter."""
        import HMF_convo
        multi_obs_convolution = HMF_convo.MultiObsConvolution(self.observable_pairs,
                                                              self.pairs_zmin, self.pairs_zmax, self.pairs_Nz,
                                                              self.surveyCutLambda, self.richness_scatter_model,
                                                              do_bias=False,
                                                              NPROC=0)
        scaling = self.test_scaling()
        dNdlnM = self.test_compute_HMF_Tinker08()
        HMF = {'z_arr': self.z_arr, 'lnM_arr': self.lnM_arr, 'dNdlnM': dNdlnM}
        dN_dmultiobs_dict = multi_obs_convolution.execute(HMF, scaling, scaling, self.cosmology)
        return dN_dmultiobs_dict

    def test_abundance(self, tmp_path_factory):
        """Poisson abundance likelihood."""
        import abundance
        SPT_survey = Table.read('data/SPT_SZ_ECS_500d_survey.txt', format='ascii.commented_header')
        f_name = tmp_path_factory.getbasetemp() / 'data0' / 'mock.fits'
        catalog = Table.read(f_name)
        number_count = abundance.NumberCount(catalog, SPT_survey,
                                             60., [.25, 1.78],
                                             NPROC=0)
        dN_dmultiobs_dict = self.test_HMF_convo()
        HMF = {'lnM_arr': dN_dmultiobs_dict['lnM_arr']}
        z = {}
        for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep', 'SZ']:
            z['%s_z' % tmp] = dN_dmultiobs_dict['%s_z' % tmp]
            HMF['%s_dNdlnM' % tmp] = dN_dmultiobs_dict[tmp]
        if np.all(z['SZ_lambdacut_shallow_z'] == z['SZ_lambdacut_deep_z']):
            HMF['z_arr'] = z['SZ_lambdacut_shallow_z']
        HMF['len_z'] = len(HMF['z_arr'])
        _ = number_count.lnlike(HMF, self.cosmology, self.scaling)
