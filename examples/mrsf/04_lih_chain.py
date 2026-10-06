#!/usr/bin/env python

'''
Embedded MRSF / EMRSF for a periodic LiH chain (Gamma point).

Five LiH units per cell; the bond of the central unit is stretched. The cell's SCF
object supplies the AO integrals (density fitted), and the response space is a window
of 9 orbitals with 12 electrons (5 doubly occupied, O1, O2, 2 virtual) around the open
shells. Lower orbitals are frozen core (they enter only through the reference density),
higher ones frozen virtuals. Localized orbitals, e.g. from AVAS, can be used the same way.
'''

import numpy
from pyscf.pbc import gto, scf, df
from pyscf import mrsf
from pyscf.data.nist import HARTREE2EV

d, R, spacing, nunit = 1.71, 2.50, 8.0, 5     # Li-H bond, central bond, unit spacing (Angstrom)
m_li, m_h = 6.941, 1.008
L = spacing * nunit
atoms = []
for k in range(nunit):
    x0 = 2.0 + k * spacing                    # center of mass of each unit
    r = R if k == nunit // 2 else d
    atoms += [['Li', (x0 - r * m_h / (m_li + m_h), 10., 10.)],
              ['H', (x0 + r * m_li / (m_li + m_h), 10., 10.)]]

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

ncore = int(numpy.count_nonzero(mf.mo_occ == 2)) - 5     # 5 doubly occupied orbitals active
nact = 9
c = mf.mo_coeff
for extended in (False, True):
    td = mrsf.TDA_MRSF(mf, mo_coeff=c[:, ncore:ncore + nact],
                             mo_occ=mf.mo_occ[ncore:ncore + nact],
                             mo_core=c[:, :ncore], extended=extended)
    td.kernel(nstates=5)
    print(f'\n{"EMRSF" if extended else "MRSF"} singlets, LiH chain (central bond {R:.2f} A)')
    print(f'  reference (triplet ROHF) {td.e_ref:14.8f} Eh')
    for i, (et, e) in enumerate(zip(td.e_tot, td.e)):
        print(f'  S{i}  {et:14.8f} Eh  {(e - td.e[0]) * HARTREE2EV:8.4f} eV')
