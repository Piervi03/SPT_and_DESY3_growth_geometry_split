import numpy as np
import pytest
from scipy.interpolate import interp1d


class TestClass:

    cosmology = {'Omega_m': .3, 'Omega_l': .7, 'Omega_b': .04, 'Omnuh2': .006,
                 'h': 0.7,
                 'w0': -1., 'wa': 0.,
                 'n_s': .96, 'ln1e10As': 3.001}

    z_arr_pk = np.linspace(0,2,21)

    observable_pairs = ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep', 'richness_SZ']
    pairs_zmin = [.25, .25, .25]
    pairs_zmax = [1.78, 1.78, 1.78]
    pairs_Nz = [154, 154, 154]

    tmp = np.loadtxt('tests/MCMF_lambda_min.txt', unpack=True)
    surveyCutLambda = {'shallow': interp1d(tmp[0], tmp[1], kind='linear'),
                       'deep': interp1d(tmp[0], tmp[2], kind='linear')}
    richness_scatter_model = 'lognormal'

    def test_init_scaling(self):
        import set_scaling
        self.scaling_setter = set_scaling.SetScaling('tests/WLsimcalib_data_Megacam.py', 'tests/HST-39_200_X.txt')

    def test_init_HMF_convo(self):
        import HMF_convo
        self.multi_obs_convolution = HMF_convo.MultiObsConvolution(self.observable_pairs,
                                                                   self.pairs_zmin, self.pairs_zmax, self.pairs_Nz,
                                                                   self.surveyCutLambda, self.richness_scatter_model,
                                                                   False,
                                                                   0)

    def test_get_pk_baccoemu(self):
        import baccoemu
        emulator = baccoemu.Matter_powerspectrum()

        self.k, self.Pk = emulator.get_linear_pk(omega_matter=self.cosmology['Omega_m'],
                                                 omega_baryon=self.cosmology['Omega_b'],
                                                 hubble=self.cosmology['h'],
                                                 ns=self.cosmology['n_s'],
                                                 w0=self.cosmology['w0'],
                                                 wa=self.cosmology['wa'],
                                                 neutrino_mass=self.cosmology['Omnuh2']*94.,
                                                 A_s=1e-10*np.exp(self.cosmology['ln1e10As']),
                                                 expfactor=1./(1.+self.z_arr_pk),
                                                 cold=True)
        
