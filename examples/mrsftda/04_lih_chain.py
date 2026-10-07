#!/usr/bin/env python

'''
Embedded MRSF / EMRSF for a periodic LiH chain (Gamma point).

Five LiH units per cell; the bond of the central unit is stretched. The cell's SCF
object supplies the AO integrals (density fitted). The response space is the AVAS
active space of the three central units (Li 2s, H 1s): here 6 orbitals with 6 electrons
(2 doubly occupied, O1, O2, 2 virtual). AVAS rotates only within the doubly occupied
and the empty orbitals (canonicalize=False) and keeps the singly occupied ones
(openshell_option=3), so its orbitals describe the same determinant as mf, ordered as
core | active | virtual with the occupations mf.mo_occ. The core and virtual orbitals
are frozen: doubly occupied ones enter only through the reference density, empty ones
are dropped.
'''

import numpy
from pyscf.pbc import gto, scf, df
from pyscf.mcscf import avas
from pyscf import mrsftda
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

labels = ['2 Li 2s', '3 H 1s', '4 Li 2s', '5 H 1s', '6 Li 2s', '7 H 1s']
ncas, nelecas, mo = avas.AVAS(mf, labels, threshold=0.5, openshell_option=3,
                              minao=cell.basis, canonicalize=False, with_iao=True).kernel()
ncore = (cell.nelectron - nelecas) // 2
frozen = list(range(ncore)) + list(range(ncore + ncas, mo.shape[1]))
for extended in (False, True):
    td = mrsftda.TDA_MRSF(mf, mo_coeff=mo, mo_occ=mf.mo_occ, frozen=frozen,
                          extended=extended)
    td.kernel(nstates=5)
    print(f'\n{"EMRSF" if extended else "MRSF"} singlets, LiH chain (central bond {R:.2f} A)')
    print(f'  reference (triplet ROHF) {td.e_ref:14.8f} Eh')
    for i, (et, e) in enumerate(zip(td.e_tot, td.e)):
        print(f'  S{i}  {et:14.8f} Eh  {(e - td.e[0]) * HARTREE2EV:8.4f} eV')
