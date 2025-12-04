#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:07:30 2025

@author: bingg
"""
import numpy as np
from scipy.linalg import eigh
from pyqed.qchem.mcscf.casci import CASCI
from opt_einsum import contract

from pyqed.qchem.optimize import minimize

class CASSCF(CASCI):
    """

    Using the OptOrbFCI algorithm to optimize orbitals
    (better than conventional CASSCF algorithm)



    """
    def run(self):
        pass

    def state_average(self, weights):
        self.nstates = len(weights)
        self.weights = weights
        pass


def energy(U, h1e, eri, dm1, dm2):
    """
    electronic energy 

    Parameters
    ----------
    U : ndarray of (n, p < n/2) 
        transformation matrix
    h1e : TYPE
        core Hamiltonian in canonical MO
    eri : TYPE
        DESCRIPTION.
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    e : TYPE
        DESCRIPTION.

    """
    
    e = contract('pq, pa, qb, ab ->', h1e, U, U, dm1)  
    e += 0.5 * (contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U, U, U, U, dm2))
    return e 

def kernel(mf, U0, max_steps=50, tol=1e-6):
    """
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    where G_k = \nabla P(U_k) is the gradient.

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        DESCRIPTION.
    U0: ndarray
        initial guess of orbitals
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    k = 0
    U = U0 # initial guess for U0

    # first FCI calculation
    mc = CASCI(mf, ncas=2, nelecas=2)
    mc.run(U)
    e_old = mc.e_tot


    for k in range(max_steps):
        
        # update CI coeff
        mc.run(U)
        
        if abs(mc.e_tot - e_old) < tol:
            print("E(CASSCF) = {}".format(mc.e_tot))
            break

        dm1, dm2 = mc.make_rdm12(0)
        h1e = mc.hcore
        eri = mc.eri_so[0, 0] # for spin-restricted calculation

        # update the MOs by updating U
        U, E = minimize(energy, U, args=(h1e, eri, dm1, dm2))

    mc.run()

    # fcisolver(h1e, )

def constrained_optimization(U, h1e, h2e, dm1, dm2, max_steps=50):
    """
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        ERI.
    dm1 : TYPE
        1RDM.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # orb opt
    converged = False
    k = 0

    # add random noise
    U += 0.1 * np.random.randn(U.shape)
    U = orth(U)

    U_old = U.copy()
    for k in range(max_steps):

        G = gradient(U, h1e, h2e, dm1, dm2)
        U = orth(U - stepsize(k) * G)

        if 1 - abs(inner(U_old, U)) < 1e-3:
            converged = True
            break

        U_old = U.copy()
        k += 1

    if converged:
        return U
    else:
        raise RuntimeError('Constrained optimization not converged.')


def gradient(U, h1e, h2e, dm1, dm2):
    g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
    g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    return g




import torch

# from expm32 import expm32, differential

def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]
