#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:32:14 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

# if __name__ == "__main__":
import numpy as np
from pyqed import Molecule
from pyqed.qchem.ci import CASCI
from pyqed.qchem.ci.casci import overlap

import copy


# mol = Molecule(atom = [
# ['Li' , (0. , 0. , 0)],
# ['F' , (0. , 0. , 1)], ])

# mol.basis = '631g'
# mol.charge = 0

# mol.molecular_frame()


# print(mol.atom_coords())

nstates = 3
Rs = np.linspace(2.8, 2.82, 2)
E = np.zeros((nstates, len(Rs)))

R = Rs[0]

atom = [
    ['Na' , (0. , 0. , 0)],
    ['F' , (0. , 0. , R)]]

mol = Molecule(atom, basis='631g')
mol.molecular_frame()

mol.build()

# S = overlap(casci_old, casci)

print(mol.atom_coords())

mf = mol.RHF()
mf.run()

dm = mf.make_rdm1()

ncas, nelecas = (4,2)
casci = CASCI(mf, ncas, nelecas)

casci.run(nstates)

E[:,0] = casci.e_tot

casci_old = copy.deepcopy(casci)

# S = overlap(casci, casci)
# print(S)

for i in range(1, len(Rs)):

    R = Rs[i]
    print('bond length = ', R)

    atom = [
        ['Na' , (0. , 0. , 0)],
        ['F' , (0. , 0. , R)]]

    mol = Molecule(atom, basis='631g')

    mol.molecular_frame()

    mol.build()

    print(mol.atom_coords())

    mf = mol.RHF()
    mf.run(dm0=dm)

    dm = mf.make_rdm1()

    # ncas, nelecas = (6,2)
    casci = CASCI(mf, ncas, nelecas)

    casci.run(nstates)

    # print(casci.binary)

    E[:,i] = casci.e_tot


    S = overlap(casci_old, casci)

    print(S)

    casci_old = copy.deepcopy(casci)


# np.savez('energy', E)

# import ultraplot as plt

# fig, ax = plt.subplots()
# for n in range(nstates):
#     ax.plot(Rs, E[n])

#### test overlap
# mol2 = Molecule(atom = [
# ['H' , (0. , 0. , 0)],
# ['H' , (0. , 0. , 1.4)], ])
# mol2.basis = '631g'

# # mol.unit = 'b'
# mol2.build()

# mf2 = mol2.RHF().run()


# ncas, nelecas = (4,2)
# casci2 = CASCI(mf2, ncas, nelecas).run(2)

# casci.run()
# S = overlap(casci, casci2)
# print(S)