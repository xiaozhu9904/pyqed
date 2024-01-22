#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 01:09:35 2024

@author: bingg
"""

import numpy as np
import pylab as pl
from scipy.linalg import expm, block_diag
import logging

from scipy.fftpack import fft, ifft, fftfreq, fftn, ifftn

from decompose import decompose, compress

from pyqed import gwp, discretize, pauli, sigmaz, interval

from pyqed.tensor.mps import MPS
from pyqed.tensor.decompose import compress


class SPO:
    def __init__(self, nstates, domains, levels, chi_max, dvr_type='sinc'):
        """
         MPS representation for LDR dynamics using the SPO integrator
         
         The first N sites are nuclear while last site is the electronic.
         modes. :math:`| \alpha n_1 n_2 \cdots n_d\rangle`
         

        Parameters
        ----------
        nstates : TYPE
            number of electronic states.
        domains : TYPE
            range for nuclear dofs
        levels : TYPE
            discretization levels for all nuclear coordinates.
        chi_max : TYPE
            bond dimension
        dvr_type : TYPE, optional
            DESCRIPTION. The default is 'sinc'.

        Returns
        -------
        None.

        """   
        
        self.ndim = len(levels)  # nuclear degrees of freedom
        
        self.nsites = self.L = self.ndim + 1 # last site electronic 
        
        if dvr_type == 'sinc': 
            logging.info('Collert-Miller DRV using particle in a box eigenstates;\
                         boundary points excluded.')

            self.x = []
            for d in range(self.ndim):
                a, b = domains[d]
                self.x.append(discretize(a, b, levels[d], endpoints=False))
            
        self.dims = [len(_x) for _x in self.x] # [B.shape[1] for B in B_list]
        self.dx = [interval(x) for x in self.x]
        self.chi_max = chi_max
        self.kx = [2. * np.pi * fftfreq(self.dims[l], self.dx[l]) for l in self.L]

        # make_V_list(X, Y)
    
        self.apes = None
        self.electronic_overlap = None 
    
    def kinetic(self, B_list, dt):
        for i in range(self.ndim):
            
            # kx = 2. * np.pi * fftfreq(self.dims[i], self.dx[i])

            _, chi1, chi2 = np.shape(B_list[i])

            # The FFT cannot be used in LDR, different from the BO dynamics.
            # B_list[i] = ifftn(np.einsum('i, aib -> aib', np.exp(-0.5j * kx**2 * dt), \
            #                             fftn(B_list[i], axes=(1))), axes=(1))
            return B_list
        
    def potential(self, B_list, v_tt, chi_max):
        
        # L = len(B_list)
        L = self.L        
        
        # U = np.exp(-1j * dt * V)
        
        # decompose the potential energy matrix
        
        # vf, vs = decompose(U, rank=chi_max)
        
        As = [] 
        for i in range(L):
            
            a1, d, a2 = v_tt[i].shape 
            chi1, d, chi2 = B_list[i].shape 
            
            A = np.einsum('aib, cid-> aci bd', v_tt[i], B_list[i])
            A = np.reshape(A, (a1 * chi1, d, a2 * chi2))
            
            As.append(A.copy())
        
        As, Ss = compress(As, chi_max=chi_max)   
        return As
        
    def run(self, psi0, dt, nt, nout=1):
        
        # v = self.apes 
        
        chi_max = self.chi_max
        
        v = self.apes
        assert(v.shape[-1] == self.nstates)
        
        V = np.exp(-1j * v * dt)

        # decompose the potential propagator        
        vf, vs = decompose(V, chi_max)
        
        X = sigmaz()
        Xs = []
        
        B_list = psi0
        
        for n in range(nt):
            for k1 in range(nout):
                
                # kinetic energy evolution

                    
                # potential energy evolution
                
                B_list, s_list = self.potential(B_list, vf, chi_max)
            
            Xs.append(self.expect_one_site(B_list, X))
            
        return Xs


if __name__=="__main__":
    
    def vibronic_state(x, nstates=2, ndim=3, dtype=complex):
        """
        Create an initial product vibronic state.
        input:
            L: number of sites
            chi_max: maximum bond dimension
            d: local dimension for each site
    
        return
        =======
        MPS in right canonical form S0-B0--B1-....B_L
        """
        B_list = []
        s_list = []
        n = len(x)
        # electronic state is placed at the end 
        # dims = [n, ] * ndim + [nstates]
        
        L = ndim + 1 # the last site represents electronic space
        
        g = gwp(x, x0=-1)
        
        for i in range(L-1):
            B = np.zeros((1, n, 1),dtype=dtype)
            B[0, :, 0] = g
            
            s = np.zeros(1)
            s[0] = 1.
            B_list.append(B)
            s_list.append(s)

        B[0, 1, 0] = 1. 
        B_list.append(B)
        
        s[0] = 1.
        s_list.append(s)
        
        # return B_list,s_list
        return MPS(B_list)
    
    # Define Pararemeter here
    delta = dt = 0.02
    L = 2
    chi_max = 10
    N_steps = 10
 
    # # grid
    d = 2**4 # local size of Hilbert space
    # x = np.linspace(-2,2,d, endpoint=False)
    # y = np.linspace(-2,2,d, endpoint=False)
    
    
    # print(interval(x))
    
    # X, Y = np.meshgrid(x,y)
 
    # V = make_V_list(X,Y)
    def pes(x):
        dim = len(x)
        v = 0 
        for d in range(dim):
            v += 0.5 *d* x[d]**2
        v += 0.3 * x[0] * x[2] #+ x[0]**2 * 0.2
        return v
    
    
    # a = np.random.randn(3, 3, 3)
    level = 4
    # n = 2**level - 1 # number of grid points for each dim
    x = np.linspace(-6, 6, 2**level, endpoint=False)[1:]
    n = len(x)
 
    dx = interval(x)
 
    
    v = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v[i, j, k] = pes([x[i], x[j], x[k]])
                
    # frequency space
    kx = 2. * np.pi * fftfreq(n, dx)
    
 
 
    # TEBD algorithm
    L = 3
    # B_list,s_list 
    
    # initialize a vibronic state
    # mps = initial_state(n, chi_max, L, dtype=complex)
    mps = vibronic_state(x)
    
    
 
    # spo = SPO(L, dims=[n, ] * 3, chi_max=6)
    # spo.set_apes(v)
    
    # Xs = spo.run(B_list, s_list, dt=0.04, nt=500)
    
    # # print(len(B_list), len(s_list))
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(Xs)
 
    # S = [0]
    # for step in range(N_steps):
 
    #     B_list = kinetic(k, B_list)
    #     B_list, s_list = potential(B_list, s_list, V, chi_max)
 
    #     s2 = np.array(s_list[L//2])**2
    #     S.append(-np.sum(s2*np.log(s2)))
 
    # pl.plot(delta*np.arange(N_steps+1),S)
    # pl.xlabel('$t$')
    # pl.ylabel('$S$')
    # pl.legend(['MPO','TEBD'],loc='upper left')
    # pl.show()