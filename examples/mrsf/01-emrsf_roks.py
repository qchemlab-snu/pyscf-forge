#!/usr/bin/env python
'''
MRSF / EMRSF singlet TDA on a triplet ROKS reference.

EMRSF adds the core->virtual configurations of the closed-shell singlet (as in
LR-TDDFT) to MRSF. Energies are relative to the triplet reference; the lowest
root is S0. Above td.dense_threshold the matrix-free Davidson solver is used.
'''

import numpy
from pyscf import gto, dft
from pyscf import mrsf

mol = gto.M(atom='''
O   0.000000000   0.000000000  -0.041061554
H  -0.533194329   0.533194329  -0.614469223
H   0.533194329  -0.533194329  -0.614469223''', basis='cc-pvdz', spin=2)
mf = dft.ROKS(mol, xc='0.5*HF+0.5*B88,LYP').run()

td = mrsf.TDA_EMRSF(mf)          # same as mf.TDA_EMRSF(); mrsf.TDA_MRSF(mf) for MRSF
td.nstates = 5
td.dense_threshold = 0           # force the matrix-free solver for this small example
td.kernel()
td.analyze()

# Validation against the explicit matrix
w = numpy.linalg.eigvalsh(td.get_ab_mrsf())[:td.nstates]   # dense A (TDA: no B)
print('max |Davidson - eigh| (Eh):', abs(td.e - w).max())
print('vertical excitations from S0 (eV):', (td.e[1:] - td.e[0]) * 27.211386245988)

# Charge-transfer case: LiF near the ionic/covalent avoided crossing
mol = gto.M(atom='Li 0 0 0; F 0 0 4.4', basis='6-31g', spin=2)
mf = dft.ROKS(mol, xc='0.76*HF+0.24*B88,LYP').set(init_guess='huckel').run()
td = mf.TDA_EMRSF()
td.nstates = 4
td.kernel()
td.analyze()
