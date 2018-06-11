from __future__ import division
import numpy as np
import imp
from cosmosis.datablock import option_section
import cosmo


class XrayProfile:
    def __init__(self, options):
        self.todo = {'Yx': options.get_bool(option_section, 'doYx'),
            'Mgas': options.get_bool(option_section, 'doMgas')}
        self.todo['Xobs'] = 'Mg' if self.todo['Mgas'] else 'Yx'
        self.YXPARAM = options.get_string(option_section, 'YXPARAM')
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        self.XraymPivot = options.get_double(option_section, 'XraymPivot')
        self.XraySample = SPTdata.XraySample

        ##### Reference cosmology for which Mgas is measured
        self.cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}


    ########################################
    def setRef(self, catalog):
        """Extract an X-ray observable from the measured profiles. Find radius r at which
            data(r)==scalingRelation(r).

        Returns: ln(likelihood) to account for the fact that the observable is actually a
            function of the parameters.

        Note: Everything in this function is in units of Mpc or Msun, no h!
        """

        catalog['XrayRef'] = [None for i in range(len(catalog['SPT_ID']))]

        for i,name in enumerate(catalog['SPT_ID']):
            ##### Exclude double entries
            if name in self.XraySample and (name,catalog['field'][i]) not in self.SPTdoubleCount:

                E_z = cosmo.Ez(catalog['redshift'][i], self.cosmologyRef)
                rho_c_z = cosmo.RHOCRIT * (self.cosmologyRef['h']*cosmo.Ez(catalog['redshift'][i], self.cosmologyRef))**2

                ##### Initial M500 guess, and observational error
                if self.todo['Xobs']=='Yx':
                    obserr = catalog['lnYx_err'][i]
                elif self.todo['Xobs']=='Mg':
                    obserr = catalog['lnMg_err'][i]

                # r500 from mass estimate
                r500ref = 1000 * (3*catalog['M500'][i]/(4*np.pi*500*rho_c_z))**(1/3)

                nonzero = np.nonzero(catalog['Mg'][i][0])[0]
                Xnew = np.interp(r500ref, catalog['Mg'][i][0,nonzero], catalog['Mg'][i][1,nonzero])

                if self.todo['Xobs']=='Yx':
                        Xnew*= 1e-14*catalog['Tx'][i]

                catalog['XrayRef'][i] = (r500ref, Xnew, obserr)
