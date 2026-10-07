#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from pyscf import gto, scf, dft, ao2mo, mrsftda
from pyscf.data import nist
from pyscf.fci import direct_spin1, cistring, spin_op
from pyscf.mrsftda import rohf_mrsf

XC_BHHLYP = 'HF*0.5+0.5*B88, 1.0*LYP'


def setUpModule():
    global mol_c2h4, mf_c2h4, mol_h2o, mf_h2o, mol, mf_hf, mf_ks
    mol_c2h4 = gto.M(atom='''C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237;
                             H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237''',
                     basis='6-31g', spin=2, verbose=0, output='/dev/null')
    mf_c2h4 = dft.ROKS(mol_c2h4, xc=XC_BHHLYP)
    mf_c2h4.grids.atom_grid = (96, 302)
    mf_c2h4.grids.prune = None
    mf_c2h4.conv_tol = 1e-11
    mf_c2h4.kernel()

    h2o = '''O 0.000000000 0.000000000 -0.041061554
             H -0.533194329 0.533194329 -0.614469223
             H 0.533194329 -0.533194329 -0.614469223'''
    mol_h2o = gto.M(atom=h2o, basis='6-31g*', cart=True, spin=2, verbose=0, output='/dev/null')
    mf_h2o = dft.ROKS(mol_h2o, xc=XC_BHHLYP)
    mf_h2o.grids.atom_grid = (96, 302)
    mf_h2o.grids.prune = None
    mf_h2o.conv_tol = 1e-11
    mf_h2o.kernel()

    mol = gto.M(atom=h2o, basis='6-31g', spin=2, verbose=0, output='/dev/null')
    mf_hf = scf.ROHF(mol)
    mf_hf.conv_tol = 1e-11
    mf_hf.kernel()
    mf_ks = dft.ROKS(mol, xc=XC_BHHLYP)
    mf_ks.grids.atom_grid = (50, 194)
    mf_ks.conv_tol = 1e-11
    mf_ks.kernel()


def tearDownModule():
    global mol_c2h4, mf_c2h4, mol_h2o, mf_h2o, mol, mf_hf, mf_ks
    for m in (mol_c2h4, mol_h2o, mol):
        m.stdout.close()
    del mol_c2h4, mf_c2h4, mol_h2o, mf_h2o, mol, mf_hf, mf_ks


def active_space(mf, nfc, nact):
    '''Frozen core, response orbitals and the active-space Hamiltonian (HF reference).'''
    c, occ = mf.mo_coeff, mf.mo_occ
    c_core, c_act, occ_act = c[:, :nfc], c[:, nfc:nfc + nact], occ[nfc:nfc + nact]
    dmc = 2 * c_core.dot(c_core.T)
    vj, vk = mf.get_jk(mf.mol, dmc)
    vcore = vj - .5 * vk
    h1e = c_act.T.dot(mf.get_hcore() + vcore).dot(c_act)
    e_core = (mf.energy_nuc() + np.einsum('ij,ji', dmc, mf.get_hcore())
              + .5 * np.einsum('ij,ji', dmc, vcore))
    return c_core, c_act, occ_act, h1e, e_core


class KnownValues(unittest.TestCase):
    # Reference energies from OpenQP 1.2.1, the implementation of the method's authors,
    # with the same geometry, basis and an unpruned (96, 302) grid.
    def test_c2h4_singlet(self):
        ref = [-4.199006, 4.155192, 4.236536, 5.194140, 5.499136, 5.602309, 6.369928]
        e = mrsftda.TDA_MRSF(mf_c2h4).kernel(nstates=9)[0]
        self.assertAlmostEqual(abs(e[:7] * nist.HARTREE2EV - ref).max(), 0, 3)

    def test_c2h4_triplet(self):
        ref = [0.768355, 3.924155, 5.000146, 5.300839, 5.351337, 6.047710, 6.886402, 8.370628]
        td = mrsftda.TDA_MRSF(mf_c2h4)
        td.singlet = False
        e = td.kernel(nstates=10)[0]
        self.assertAlmostEqual(abs(e[:8] * nist.HARTREE2EV - ref).max(), 0, 3)

    def test_h2o_singlet(self):
        # OpenQP example H2O_BHHLYP-MRSFTDDFT_ENERGY (Cartesian d functions), Hartree, and
        # its oscillator strengths S0 -> S1..S4
        ref = [-0.28348967234148004, 0.045304512523271154, 0.10356631483242085]
        td = mrsftda.TDA_MRSF(mf_h2o)
        e = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(e[:3] - ref).max(), 0, 4)
        self.assertAlmostEqual(abs(td.oscillator_strength() - [0.0140, 0., 0.1213, 0.2826]).max(), 0, 3)

    def test_dense_vs_davidson(self):
        for mf in (mf_hf, mf_ks):
            for extended in (False, True):
                for singlet in (True, False):
                    td = mrsftda.TDA_MRSF(mf, extended=extended)
                    td.singlet = singlet
                    e = td.kernel(nstates=5)[0]
                    td.dense_threshold = 0
                    self.assertAlmostEqual(abs(td.kernel(nstates=5)[0] - e).max(), 0, 8)

    def test_spin_pairing_coupling_vs_fci(self):
        # A' is the Hamiltonian element between an MS = +1 and an MS = -1 spin-flipped
        # determinant; compare magnitudes with the element from pyscf.fci
        nc, nv = 3, 3
        norb = nc + 2 + nv
        o1, o2 = nc, nc + 1
        cidx, oidx, vidx = np.arange(nc), np.array([o1, o2]), np.arange(nc + 2, norb)
        eri = np.random.default_rng(7).standard_normal((norb,) * 4)
        eri = eri + eri.transpose(1, 0, 2, 3)
        eri = eri + eri.transpose(0, 1, 3, 2)
        eri = (eri + eri.transpose(2, 3, 0, 1)) / 8
        kmo = dict(((a, b), eri[:, nc + a, :, nc + b]) for a in range(2) for b in range(2))
        cpl = rohf_mrsf.spin_pair_block(kmo, 1., cidx, oidx, vidx)
        ncv = nc * nv
        pairs = [(m, a) for m in (o1, o2) for a in vidx] + [(i, m) for i in cidx for m in (o1, o2)]
        core = list(cidx)
        n = nc + 1
        h2e = direct_spin1.absorb_h1e(np.zeros((norb, norb)), eri, norb, (n, n), .5)

        def addr(occ):
            return cistring.str2addr(norb, n, sum(1 << int(k) for k in occ))
        for i, (p, q) in enumerate(pairs):
            ci0 = np.zeros((cistring.num_strings(norb, n),) * 2)
            ci0[addr([x for x in core + [o1, o2] if x != p]), addr(core + [q])] = 1
            hci = direct_spin1.contract_2e(h2e, ci0, norb, (n, n))
            for j, (r, s) in enumerate(pairs):
                ref = hci[addr(core + [s]), addr([x for x in core + [o1, o2] if x != r])]
                self.assertAlmostEqual(abs(cpl[ncv + i, ncv + j]), abs(ref), 12)

    def test_emrsf_coupling_vs_fci(self):
        # D block between the singlet OV(O2->v) / CO(c->O1) configurations and the CV_ext
        # configurations (c->v on the closed-shell O1^2 reference): exact Hamiltonian
        # elements between the singlet configuration functions (HF reference)
        nc, nv = 2, 2
        nfc = int((mf_hf.mo_occ == 2).sum()) - nc
        norb = nc + 2 + nv
        c_core, c_act, occ_act, h1e, e_core = active_space(mf_hf, nfc, norb)
        eri = ao2mo.restore(1, ao2mo.full(mol, c_act), norb)
        a = mrsftda.TDA_MRSF.given_mo_ints(h1e, eri, occ_act, e_core=e_core, extended=True).get_ab()
        ncv = nc * nv
        nm = 3 + ncv + 2 * nv + 2 * nc
        C, O1, O2, V = [0, 1], 2, 3, [4, 5]
        nel = (3, 3)
        h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nel, .5)
        na = cistring.num_strings(norb, 3)

        def det(alpha, beta):
            v = np.zeros((na, na))
            v[cistring.str2addr(norb, 3, sum(1 << i for i in alpha)),
              cistring.str2addr(norb, 3, sum(1 << i for i in beta))] = 1
            return v

        def singlet(d1, d2):
            for sgn in (1, -1):
                v = (d1 + sgn * d2) / np.sqrt(2)
                if abs(spin_op.spin_square0(v, norb, nel)[0]) < 1e-10:
                    return v

        def cvx(c, v):
            occ = [x for x in C + [O1] if x != c] + [v]
            return singlet(det(occ, C + [O1]), det(C + [O1], occ))
        rows = [(3 + ncv + nv + iv, singlet(det(C + [O1], C + [V[iv]]), det(C + [V[iv]], C + [O1])))
                for iv in range(nv)]                                            # OV(O2 -> v)
        for ic in range(nc):                                                    # CO(c -> O1)
            oc = [x for x in C if x != C[ic]]
            rows.append((3 + ncv + 2 * nv + 2 * ic,
                         singlet(det(oc + [O1, O2], C + [O1]), det(C + [O1], oc + [O1, O2]))))
        for i, bra in rows:
            for ic in range(nc):
                for iv in range(nv):
                    ket = cvx(C[ic], V[iv])
                    ref = np.vdot(bra, direct_spin1.contract_2e(h2e, ket, norb, nel))
                    self.assertAlmostEqual(abs(a[i, nm + ic * nv + iv]), abs(ref), 9)

    def test_frozen_orbitals_and_integrals(self):
        # embedding: frozen core + frozen virtuals of the SCF object, the same problem from
        # active-space integrals, and with the Fock matrices supplied
        nfc, nact = 1, 9
        c_core, c_act, occ_act, h1e, e_core = active_space(mf_hf, nfc, nact)
        frozen = list(range(nfc)) + list(range(nfc + nact, mf_hf.mo_occ.size))
        eri = ao2mo.restore(1, ao2mo.full(mol, c_act), nact)
        fock = rohf_mrsf.mf_embedded(mf_hf, c_act, occ_act, c_core)[1:3]
        fock_cv = rohf_mrsf.cv_proxy(mf_hf, c_act, occ_act, c_core)[1]
        for extended in (False, True):
            for singlet in (True, False):
                ref = mrsftda.TDA_MRSF(mf_hf, frozen=frozen, extended=extended)
                ref.singlet = singlet
                e0 = ref.kernel(nstates=5)[0]
                for td in (mrsftda.TDA_MRSF.given_mo_ints(h1e, eri, occ_act, e_core=e_core,
                                                          extended=extended),
                           mrsftda.TDA_MRSF(mf_hf, frozen=frozen, extended=extended,
                                            fock=fock, fock_cv=fock_cv)):
                    td.singlet = singlet
                    self.assertAlmostEqual(abs(td.kernel(nstates=5)[0] - e0).max(), 0, 9)
                    self.assertAlmostEqual(td.e_ref, ref.e_ref, 9)

    def test_frozen_rotated_orbitals(self):
        # non-canonical orbitals (rotated within the frozen core, the response doubly
        # occupied, the response empty and the frozen empty spaces) in shuffled order give
        # the same states: the Fock matrices are rebuilt from the orbitals
        nfc, nact = 2, 9
        c, occ = mf_hf.mo_coeff, mf_hf.mo_occ
        nmo = occ.size
        frozen = list(range(nfc)) + list(range(nfc + nact, nmo))
        rng = np.random.default_rng(5)
        c_rot = c.copy()
        nd = int((occ == 2).sum())
        for a, b in ((0, nfc), (nfc, nd), (nd + 2, nfc + nact), (nfc + nact, nmo)):
            u = np.linalg.qr(rng.standard_normal((b - a, b - a)))[0]
            c_rot[:, a:b] = c[:, a:b].dot(u)
        perm = rng.permutation(nmo)
        perm[np.sort(np.where((perm == nd) | (perm == nd + 1))[0])] = nd, nd + 1  # O1 before O2
        frozen_perm = [int(np.where(perm == k)[0][0]) for k in frozen]
        for extended in (False, True):
            ref = mrsftda.TDA_MRSF(mf_hf, frozen=frozen, extended=extended)
            e0 = ref.kernel(nstates=5)[0]
            f0 = ref.oscillator_strength()
            td = mrsftda.TDA_MRSF(mf_hf, c_rot[:, perm], occ[perm], frozen=frozen_perm,
                                  extended=extended)
            self.assertAlmostEqual(abs(td.kernel(nstates=5)[0] - e0).max(), 0, 9)
            self.assertAlmostEqual(td.e_ref, ref.e_ref, 9)
            self.assertAlmostEqual(abs(td.oscillator_strength() - f0).max(), 0, 7)

    def test_ks_fock_given_mo_ints(self):
        # Kohn-Sham Fock matrices + orbital integrals + hyb reproduce MRSF of the KS reference
        c, occ = mf_ks.mo_coeff, mf_ks.mo_occ
        nmo = c.shape[1]
        h1e = c.T.dot(mf_ks.get_hcore()).dot(c)
        eri = ao2mo.restore(1, ao2mo.full(mol, c), nmo)
        fock = rohf_mrsf.mf_embedded(mf_ks, c, occ)[1:3]
        hyb = rohf_mrsf.hybrid_coeff(mf_ks)
        for singlet in (True, False):
            ref = mrsftda.TDA_MRSF(mf_ks)
            ref.singlet = singlet
            e0 = ref.kernel(nstates=5)[0]
            td = mrsftda.TDA_MRSF.given_mo_ints(h1e, eri, occ, e_core=mf_ks.energy_nuc(), fock=fock,
                                                hyb=hyb, e_ref=mf_ks.e_tot)
            td.singlet = singlet
            self.assertAlmostEqual(abs(td.kernel(nstates=5)[0] - e0).max(), 0, 9)
            self.assertAlmostEqual(td.e_ref, mf_ks.e_tot, 12)
        with self.assertRaises(NotImplementedError):     # EMRSF needs the XC kernel of mf
            mrsftda.TDA_MRSF.given_mo_ints(h1e, eri, occ, fock=fock, hyb=hyb, extended=True).kernel()

    def test_pbc_gamma(self):
        from pyscf.pbc import gto as pgto, scf as pscf, df as pdf
        cell = pgto.Cell()
        cell.atom = [['H', (0.75 * i, 0, 0)] for i in range(6)]
        cell.a = np.diag([4.5, 12., 12.])
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.spin = 2
        cell.verbose = 0
        cell.build()
        kmf = pscf.ROHF(cell)
        kmf.with_df = pdf.GDF(cell)
        kmf.exxdiv = None
        kmf.conv_tol = 1e-11
        kmf.kernel()
        nfc = int((kmf.mo_occ == 2).sum()) - 1
        c_core, c_act, occ_act, h1e, e_core = active_space(kmf, nfc, 4)
        eri = ao2mo.restore(1, kmf.with_df.ao2mo(c_act), 4)
        frozen = list(range(nfc)) + list(range(nfc + 4, kmf.mo_occ.size))
        for extended in (False, True):
            td = mrsftda.TDA_MRSF(kmf, frozen=frozen, extended=extended)
            e = td.kernel(nstates=3)[0]
            if extended:
                td.d_mo = False
                self.assertAlmostEqual(abs(td.kernel(nstates=3)[0] - e).max(), 0, 9)
            ti = mrsftda.TDA_MRSF.given_mo_ints(h1e, eri, occ_act, e_core=e_core, extended=extended)
            self.assertAlmostEqual(abs(ti.kernel(nstates=3)[0] - e).max(), 0, 9)
            self.assertAlmostEqual(ti.e_ref, td.e_ref, 9)

    def test_with_df(self):
        for extended in (False, True):
            e0 = mrsftda.TDA_MRSF(mf_ks, extended=extended).kernel(nstates=5)[0]
            td = mrsftda.TDA_MRSF(mf_ks, extended=extended)
            td.with_df = True
            e = td.kernel(nstates=5)[0]
            self.assertLess(abs(e - e0).max(), 2e-4)
            td.dense_threshold = 0
            self.assertAlmostEqual(abs(td.kernel(nstates=5)[0] - e).max(), 0, 8)

    def test_d_mo(self):
        # EMRSF D coupling from MO integrals = from AO J/K (frozen core, DF, given_mo_ints)
        nfc, nact = 1, 9
        c_core, c_act, occ_act, h1e, e_core = active_space(mf_hf, nfc, nact)
        frozen = list(range(nfc)) + list(range(nfc + nact, mf_hf.mo_occ.size))
        eri = ao2mo.restore(1, ao2mo.full(mol, c_act), nact)
        tds = [lambda: mrsftda.TDA_MRSF(mf_ks, extended=True),
               lambda: mrsftda.TDA_MRSF(mf_hf, frozen=frozen, extended=True),
               lambda: mrsftda.TDA_MRSF(mf_ks, extended=True).set(with_df=True),
               lambda: mrsftda.TDA_MRSF.given_mo_ints(h1e, eri, occ_act, e_core=e_core,
                                                      extended=True)]
        for make in tds:
            for singlet in (True, False):
                a = []
                for d_mo in (True, False):
                    td = make()
                    td.singlet = singlet
                    td.d_mo = d_mo
                    a.append(td.get_ab())
                self.assertAlmostEqual(abs(a[0] - a[1]).max(), 0, 10)

    def test_orbital_order(self):
        # orbitals not ordered as doubly occupied, open, empty (e.g. after MOM)
        td = mrsftda.TDA_MRSF(mf_h2o)
        e0 = td.kernel(nstates=5)[0]
        f0 = td.oscillator_strength()
        perm = np.random.default_rng(3).permutation(mf_h2o.mo_occ.size)
        td = mrsftda.TDA_MRSF(mf_h2o, mf_h2o.mo_coeff[:, perm], mf_h2o.mo_occ[perm])
        self.assertAlmostEqual(abs(td.kernel(nstates=5)[0] - e0).max(), 0, 9)
        self.assertAlmostEqual(abs(td.oscillator_strength() - f0).max(), 0, 7)

    def test_method_registered(self):
        self.assertTrue(isinstance(mf_h2o.TDA_MRSF(), mrsftda.TDA_MRSF))
        self.assertTrue(mf_h2o.TDA_MRSF(extended=True).extended)

    def test_invalid_reference(self):
        h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        with self.assertRaises(ValueError):
            mrsftda.TDA_MRSF(scf.ROHF(h2).run()).kernel()
        with self.assertRaises(ValueError):
            mrsftda.TDA_MRSF(dft.UKS(mol, xc=XC_BHHLYP).run()).kernel()
        with self.assertRaises(NotImplementedError):
            mrsftda.TDA_MRSF(dft.ROKS(mol, xc='CAMB3LYP').run()).kernel()
        with self.assertRaises(ValueError):     # an open-shell orbital frozen
            mrsftda.TDA_MRSF(mf_hf, frozen=[int(np.where(mf_hf.mo_occ == 1)[0][0])]).kernel()


if __name__ == '__main__':
    print('Full tests for MRSF/EMRSF-TDA')
    unittest.main()
