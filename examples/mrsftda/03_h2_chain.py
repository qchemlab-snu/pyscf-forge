#!/usr/bin/env python

'''
Embedded MRSF / EMRSF for a periodic H2 chain (Gamma point).

Five H2 units per cell; the bond of the central unit is stretched. The cell's SCF
object supplies the AO integrals (density fitted), and the response space is a window
of 6 orbitals (2 doubly occupied, O1, O2, 2 virtual) around the open shells. Lower
orbitals are frozen core (they enter only through the reference density), higher ones
frozen virtuals. Localized orbitals, e.g. from AVAS, can be given as mo_coeff and
frozen the same way.
'''

import numpy
from pyscf.pbc import gto, scf, df
from pyscf import mrsftda
from pyscf.data.nist import HARTREE2EV

d, R, spacing, nunit = 0.86, 1.50, 5.0, 5     # H-H bond, central bond, unit spacing (Angstrom)
L = spacing * nunit
atoms = []
for k in range(nunit):
    x0 = L / 2 + (k - nunit // 2) * spacing
    r = R if k == nunit // 2 else d
    atoms += [['H', (x0 - r / 2, 10., 10.)], ['H', (x0 + r / 2, 10., 10.)]]

cell = gto.Cell()
cell.atom = atoms
cell.a = numpy.diag([L, 20., 20.])
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.spin = 2
cell.verbose = 0
cell.build()

mf = scf.ROHF(cell)
mf.with_df = df.GDF(cell)
mf.exxdiv = None
mf.kernel()

ncore = int(numpy.count_nonzero(mf.mo_occ == 2)) - 2     # 2 doubly occupied orbitals active
nact = 6
frozen = list(range(ncore)) + list(range(ncore + nact, mf.mo_occ.size))
for extended in (False, True):
    td = mrsftda.TDA_MRSF(mf, frozen=frozen, extended=extended)
    td.kernel(nstates=5)
    print(f'\n{"EMRSF" if extended else "MRSF"} singlets, H2 chain (central bond {R:.2f} A)')
    print(f'  reference (triplet ROHF) {td.e_ref:14.8f} Eh')
    for i, (et, e) in enumerate(zip(td.e_tot, td.e)):
        print(f'  S{i}  {et:14.8f} Eh  {(e - td.e[0]) * HARTREE2EV:8.4f} eV')
