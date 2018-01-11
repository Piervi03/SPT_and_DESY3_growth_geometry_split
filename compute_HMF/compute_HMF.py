from __future__ import division
import numpy as np
from scipy.interpolate import interp1d

class HMFCalculator:
    def __init__(self, Deltacrit):
        self.Deltacrit = Deltacrit

        # Initialize Tinker interpolation
        x = np.log((200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.))
        # A
        y = (1.858659e-01, 1.995973e-01, 2.115659e-01, 2.184113e-01, 2.480968e-01, 2.546053e-01, 2.600000e-01, 2.600000e-01, 2.600000e-01)
        self.f1 = interp1d(x, y, kind='cubic')
        # a
        y = (1.466904e+00, 1.521782e+00, 1.559186e+00, 1.614585e+00, 1.869936e+00, 2.128056e+00, 2.301275e+00, 2.529241e+00, 2.661983e+00)
        self.f2 = interp1d(x, y, kind='cubic')
        # b
        y = (2.571104e+00, 2.254217e+00, 2.048674e+00, 1.869559e+00, 1.588649e+00, 1.507134e+00, 1.464374e+00, 1.436827e+00, 1.405210e+00)
        self.f3 = interp1d(x, y, kind='cubic')
        # c
        y = (1.193958e+00, 1.270316e+00, 1.335191e+00, 1.446266e+00, 1.581345e+00, 1.795050e+00, 1.965613e+00, 2.237466e+00, 2.439729e+00)
        self.f4 = interp1d(x, y, kind='cubic')


    def compute_HMF(self):
        return 1




    def Tinker_params(self, z, Deltamean):
        if Deltamean>3200:
            z0params = [.26, 2.66, 1.41, 2.44]
        else:
            logDeltamean = np.log(Deltamean)
            z0params = [self.f1(logDeltamean), self.f2(logDeltamean), self.f3(logDeltamean), self.f4(logDeltamean)]
        logalpha = -(.75/np.log10(Deltamean/75.))**1.2
        alpha = 10.**logalpha

        self.A = z0params[0] * (1.+z)**-.14
        self.a = z0params[1] * (1.+z)**-.06
        self.b = z0params[2] * (1.+z)**-alpha
        self.c = z0params[3] * 1.
