#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:36:15 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed import Molecule
from pyqed.qchem.mcscf import CASCI 

mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='b', basis='631g')
# mol.build()
mol = mol.topyscf()

mf = mol.RHF().run()


from pyscf.mcscf import CASSCF

mc = CASSCF(mf, ncas=2, nelecas=2)
mc.run()