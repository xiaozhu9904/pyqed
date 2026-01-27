#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:37:35 2026

@author: gugroup
"""

from pyqed.nrg import star_to_chain
from pyqed import pauli 

class SBM:
    """
    spin-boson model
    """
    def __init__(self, epsilon, Delta, omegac=1):
        """


        Parameters
        ----------
        epsilon : TYPE
            DESCRIPTION.
        Delta : TYPE
            DESCRIPTION.
        omegac : TYPE, optional
            cutoff frequency. The default is 1.

        Returns
        -------
        None.

        """

        self.omegac = omegac

        I, X, Y, Z = pauli()

        self.H = 0.5 * (- epsilon * Z + X * Delta)

    def spectral_density(self, s=1, alpha=1):
        pass

    def discretize(self):
        pass

    def to_wilson_chain(self):
        pass

    def HEOM(self):
        pass

    def Redfield(self):
        pass
    
    def DMRG(self):
        # build MPO 
        pass
