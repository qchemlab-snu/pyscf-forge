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
#
# Author: Minseok Oh <msjeff2001@snu.ac.kr>
#
'''Tests of the matrix-free EMRSF operator against the stage-2 dense matrix.'''

import io
import os
import sys
import unittest
import numpy
from pyscf import gto, scf, dft

from pyscf.mrsf import rohf_mrsf

# no point-group symmetry on purpose, so that no matrix element vanishes by symmetry
GEOM = 'O 0.05 0.02 -0.04; H -0.60 0.50 -0.62; H 0.51 -0.55 -0.58'
XC = '0.5*HF+0.5*B88,LYP'
XC_CV = '0.2*HF + 0.80*Slater + 0.72*B88, 0.81*LYP + 0.19*VWN'
TOL = 1e-10


def setUpModule():
    global mol, mf_hf, mf_gga, mf_lda, mol_dz, mf_dz, mf_df
    mol = gto.M(atom=GEOM, basis='631g', spin=2, verbose=0)
    mf_hf = scf.ROHF(mol).run(conv_tol=1e-12)
    mf_gga = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).run()
    mf_lda = dft.ROKS(mol).set(xc='0.5*HF+0.5*SLATER,VWN', conv_tol=1e-12).run()
    mol_dz = gto.M(atom=GEOM, basis='ccpvdz', spin=2, verbose=0)
    mf_dz = dft.ROKS(mol_dz).set(xc=XC, conv_tol=1e-12).run()
    mf_df = dft.ROKS(mol_dz).set(xc=XC, conv_tol=1e-12).density_fit(auxbasis='cc-pvdz-jkfit').run()


def tearDownModule():
    global mol, mf_hf, mf_gga, mf_lda, mol_dz, mf_dz, mf_df
    del mol, mf_hf, mf_gga, mf_lda, mol_dz, mf_dz, mf_df


def hfx_of(mf):
    return rohf_mrsf.TDA_EMRSF(mf).get_hfx()


def all_mfs():
    return [('HF', mf_hf), ('GGA', mf_gga), ('LDA', mf_lda)]


def op_matrix(vind, n):
    return numpy.asarray(vind(numpy.eye(n)))


class KnownValues_Proxy(unittest.TestCase):
    def test_cv_proxy_matches_cv_block(self):
        for name, mf in all_mfs():
            cidx, oidx, vidx = rohf_mrsf.orb_indices(mf)
            f_block = rohf_mrsf.cv_block(mf, None, rohf_mrsf.mo_eri(mf), cidx, oidx, vidx)[1]
            mf_cv, f_cv, hyb = rohf_mrsf.cv_proxy(mf)
            self.assertAlmostEqual(abs(f_cv - f_block).max(), 0, delta=TOL, msg=name)
            self.assertEqual(mf_cv.mol.spin, 0)
            self.assertEqual(mf_cv.mo_occ[oidx[0]], 2)
            self.assertEqual(mf_cv.mo_occ[oidx[1]], 0)
            self.assertAlmostEqual(hyb, hfx_of(mf), delta=1e-14)

    def test_cv_proxy_shares_df(self):
        mf_cv = rohf_mrsf.cv_proxy(mf_df)[0]
        self.assertIs(mf_cv.with_df, mf_df.with_df)
        self.assertIsNone(getattr(rohf_mrsf.cv_proxy(mf_dz)[0], 'with_df', None))

    def test_cv_proxy_xc_cv_on_hf_raises(self):
        self.assertRaises(ValueError, rohf_mrsf.cv_proxy, mf_hf, 'b3lyp')

    def test_unrun_scf_message(self):
        with self.assertRaises(ValueError) as ctx:
            rohf_mrsf.orb_indices(dft.ROKS(mol))
        self.assertIn('run the SCF first', str(ctx.exception))


class KnownValues_MRSFHop(unittest.TestCase):
    def test_mrsf_operator_equals_dense(self):
        from pyscf.mrsf import mrsf_hop
        for name, mf in all_mfs():
            hfx = hfx_of(mf)
            for spc in (True, False):
                a = rohf_mrsf.build_matrix(mf, hfx, expansion=False, spc=spc)
                vind, hd = mrsf_hop.gen_hop(mf, hfx, expansion=False, spc=spc)
                b = op_matrix(vind, a.shape[0])
                self.assertAlmostEqual(abs(b - a).max(), 0, delta=TOL, msg=(name, spc))
                self.assertAlmostEqual(abs(hd[:3] - a.diagonal()[:3]).max(), 0, delta=TOL, msg=name)

    def test_spin_pair_block_equals_stage2(self):
        from pyscf.mrsf import mrsf_hop
        for name, mf in [('HF', mf_hf), ('GGA', mf_gga)]:
            cidx, oidx, vidx = rohf_mrsf.orb_indices(mf)
            kmo = mrsf_hop._open_jk(mf, oidx)[1]
            c = mrsf_hop.spin_pair_block(kmo, 0.7, cidx, oidx, vidx)
            c0 = rohf_mrsf.spin_pair_coupling(rohf_mrsf.mo_eri(mf), 0.7, cidx, oidx, vidx)
            self.assertAlmostEqual(abs(c - c0).max(), 0, delta=1e-12, msg=name)

    def test_rejects_hfx_override(self):
        from pyscf.mrsf import mrsf_hop
        self.assertRaises(NotImplementedError, mrsf_hop.gen_hop, mf_gga, 0.3, False)

    def test_rejects_rsh(self):
        from pyscf.mrsf import mrsf_hop
        mf = dft.ROKS(mol).set(xc='camb3lyp', conv_tol=1e-9).run()
        self.assertRaises(NotImplementedError, mrsf_hop.gen_hop, mf, hfx_of(mf), False)

    def test_missing_sftda_message(self):
        from pyscf.mrsf import mrsf_hop
        saved = sys.modules.get('pyscf.sftda.uhf_sf')
        sys.modules['pyscf.sftda.uhf_sf'] = None
        try:
            with self.assertRaises(ImportError) as ctx:
                mrsf_hop._sf_vind(mf_gga)
            self.assertIn('pyscf-forge', str(ctx.exception))
        finally:
            if saved is None:
                del sys.modules['pyscf.sftda.uhf_sf']
            else:
                sys.modules['pyscf.sftda.uhf_sf'] = saved



class KnownValues_EMRSFHop(unittest.TestCase):
    def test_emrsf_operator_equals_dense(self):
        from pyscf.mrsf import mrsf_hop
        cases = [(n, mf, None) for n, mf in all_mfs()] + [('GGA+xc_cv', mf_gga, XC_CV)]
        for name, mf, xc_cv in cases:
            hfx = hfx_of(mf)
            for spc in (True, False):
                a = rohf_mrsf.build_matrix(mf, hfx, expansion=True, spc=spc, xc_cv=xc_cv)
                vind, hd = mrsf_hop.gen_hop(mf, hfx, expansion=True, spc=spc, xc_cv=xc_cv)
                b = op_matrix(vind, a.shape[0])
                self.assertEqual(hd.size, a.shape[0])
                self.assertAlmostEqual(abs(b - a).max(), 0, delta=TOL, msg=(name, spc))

    def test_df_operator_symmetric_and_close(self):
        from pyscf.mrsf import mrsf_hop
        vind, hd = mrsf_hop.gen_hop(mf_df, 0.5, expansion=True)
        b = op_matrix(vind, hd.size)
        self.assertAlmostEqual(abs(b - b.T).max(), 0, delta=TOL)
        w_df = numpy.linalg.eigvalsh(b)[:5]
        w = numpy.linalg.eigvalsh(rohf_mrsf.build_matrix(mf_dz, 0.5, expansion=True))[:5]
        self.assertAlmostEqual(abs(w_df - w).max(), 0, delta=5e-4)



class KnownValues_Class(unittest.TestCase):
    def test_dim_matches_dense(self):
        for expansion in (False, True):
            td = rohf_mrsf.TDA_EMRSF(mf_gga, expansion=expansion)
            self.assertEqual(td.dim, td.get_ab().shape[0])

    def test_matrix_free_kernel_matches_eigh(self):
        w = numpy.linalg.eigvalsh(rohf_mrsf.build_matrix(mf_dz, 0.5))[:5]
        td = rohf_mrsf.TDA_EMRSF(mf_dz)
        td.nstates = 5
        td.dense_threshold = 0
        e_mf, xy = td.kernel()
        self.assertIsNone(td._a)          # the dense matrix was never built
        self.assertTrue(all(td.converged))
        self.assertAlmostEqual(abs(e_mf - w).max(), 0, delta=1e-8)
        self.assertEqual(len(xy[0][0]), td.dim)

    def test_df_kernel_runs(self):
        td = rohf_mrsf.TDA_EMRSF(mf_df)
        td.nstates = 3
        td.dense_threshold = 0
        e = td.kernel()[0]
        self.assertIsNone(td._a)
        self.assertTrue(all(td.converged))
        w = numpy.linalg.eigvalsh(rohf_mrsf.build_matrix(mf_dz, 0.5))[:3]
        self.assertAlmostEqual(abs(e - w).max(), 0, delta=5e-4)

    def test_kernel_matrix_free_errors(self):
        td = rohf_mrsf.TDA_EMRSF(mf_dz)
        td.dense_threshold = 0
        td.hfx = 0.3
        self.assertRaises(NotImplementedError, td.kernel)

    def test_check_sanity_warns_on_typo(self):
        td = rohf_mrsf.TDA_EMRSF(mf_gga)
        td.stdout = io.StringIO()
        td.verbose = 3
        td.check_sanity()
        self.assertEqual(td.stdout.getvalue(), '')
        td.nstate = 5
        td.check_sanity()
        self.assertIn('nstate', td.stdout.getvalue())

    def test_init_guess_keeps_degenerate(self):
        td = rohf_mrsf.TDA_EMRSF(mf_gga)
        x0 = td.init_guess(2, numpy.array([0., 1., 1., 1., 2., 3.]))
        self.assertEqual(x0.shape, (4, 6))
        self.assertEqual(sorted(numpy.argmax(x0, axis=1).tolist()), [0, 1, 2, 3])



class KnownValues_ReviewFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol_lif = gto.M(atom='Li 0 0 0; F 0 0 4.4', basis='631g', spin=2, verbose=0)
        cls.mf_lif = dft.ROKS(mol_lif).set(xc='0.76*HF+0.24*B88,LYP', init_guess='huckel',
                                           conv_tol=1e-12).run()
        cls.mf_df631 = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).density_fit(
            auxbasis='cc-pvdz-jkfit').run()

    def test_lif_davidson_finds_lowest_roots(self):
        mf = self.mf_lif
        w = numpy.linalg.eigvalsh(rohf_mrsf.build_matrix(mf, 0.76))[:5]
        td = rohf_mrsf.TDA_EMRSF(mf)
        td.nstates = 5
        td.dense_threshold = 0
        e = td.kernel()[0]
        self.assertTrue(all(td.converged))
        self.assertAlmostEqual(abs(e - w).max(), 0, delta=1e-8)

    def test_hdiag_close_to_exact(self):
        # the diagonal integrals come from density fitting (O(N^4)), so hdiag matches the
        # exact diagonal to the DF error; it only drives the initial guess and preconditioner
        from pyscf.mrsf import mrsf_hop
        for name, mf in [('HF', mf_hf), ('GGA', mf_gga)]:
            hfx = hfx_of(mf)
            a = rohf_mrsf.build_matrix(mf, hfx, expansion=True)
            hd = mrsf_hop.gen_hop(mf, hfx, expansion=True)[1]
            nm = rohf_mrsf.TDA_EMRSF(mf, expansion=False).dim
            self.assertAlmostEqual(abs(hd[:nm] - a.diagonal()[:nm]).max(), 0, delta=2e-3, msg=name)
            if name == 'HF':   # no xc kernel: the CV diagonal is DF-close too
                self.assertAlmostEqual(abs(hd - a.diagonal()).max(), 0, delta=2e-3)

    def test_occ_diag_is_df_not_per_orbital_jk(self):
        from pyscf.mrsf import mrsf_hop
        mf = mf_gga.copy()

        def no_jk(*args, **kwargs):
            raise AssertionError('occ_diag must not build one J/K per occupied orbital')
        mf.get_jk = no_jk
        nocc = len(rohf_mrsf.orb_indices(mf)[0]) + 2
        jdiag, kdiag = mrsf_hop._occ_diag(mf, nocc)
        c = mf_gga.mo_coeff
        vj, vk = mf_gga.get_jk(mf_gga.mol, numpy.einsum('pi,qi->ipq', c[:, :nocc], c[:, :nocc]))
        j_ref = numpy.einsum('ipq,px,qx->ix', vj, c, c)
        k_ref = numpy.einsum('ipq,px,qx->ix', vk, c, c)
        self.assertAlmostEqual(abs(jdiag - j_ref).max(), 0, delta=2e-3)
        self.assertAlmostEqual(abs(kdiag - k_ref).max(), 0, delta=2e-3)

    def test_df_dense_path_equals_matrix_free(self):
        from pyscf.mrsf import mrsf_hop
        mf = self.mf_df631
        td = rohf_mrsf.TDA_EMRSF(mf)
        td.nstates = 4
        b = op_matrix(mrsf_hop.gen_hop(mf, 0.5)[0], td.dim)
        self.assertAlmostEqual(abs(td.get_ab() - b).max(), 0, delta=TOL)
        e_dense = td.kernel()[0].copy()
        td2 = rohf_mrsf.TDA_EMRSF(mf)
        td2.nstates = 4
        td2.dense_threshold = 0
        self.assertAlmostEqual(abs(td2.kernel()[0] - e_dense).max(), 0, delta=1e-8)

    def test_d_block_pieces(self):
        from pyscf.mrsf import mrsf_hop
        mf = mf_gga
        cidx, oidx, vidx = rohf_mrsf.orb_indices(mf)
        focka, fockb = rohf_mrsf.mo_fock(mf)
        jmo, kmo = mrsf_hop._open_jk(mf, oidx)
        a = rohf_mrsf.build_matrix(mf, 0.5, expansion=True)
        nm = rohf_mrsf.TDA_EMRSF(mf, expansion=False).dim
        ncv = a.shape[0] - nm
        d_parts = mrsf_hop.ext_hop(mf, 0.5, None, a[0, 0], focka, fockb, jmo, kmo,
                                    cidx, oidx, vidx)[0]
        dym = d_parts(numpy.zeros((ncv, nm)), numpy.eye(ncv))[0]
        dye = d_parts(numpy.eye(nm), numpy.zeros((nm, ncv)))[1]
        self.assertAlmostEqual(abs(dym.T - a[:nm, nm:]).max(), 0, delta=TOL)
        self.assertAlmostEqual(abs(dye.T - a[nm:, :nm]).max(), 0, delta=TOL)

    def test_cv_proxy_shares_incore_eri_and_direct_setting(self):
        self.assertIsNotNone(mf_gga._eri)
        self.assertIs(rohf_mrsf.cv_proxy(mf_gga)[0]._eri, mf_gga._eri)
        mf = dft.ROKS(mol).set(xc=XC, conv_tol=1e-10)
        mf._is_mem_enough = lambda: False
        mf.run()
        self.assertIsNone(mf._eri)
        mf_cv = rohf_mrsf.cv_proxy(mf)[0]
        self.assertFalse(mf_cv._is_mem_enough())

    def test_gen_hop_missing_sftda(self):
        from pyscf.mrsf import mrsf_hop
        saved = sys.modules.get('pyscf.sftda.uhf_sf')
        sys.modules['pyscf.sftda.uhf_sf'] = None
        try:
            self.assertRaises(ImportError, mrsf_hop.gen_hop, mf_gga, 0.5)
        finally:
            if saved is None:
                del sys.modules['pyscf.sftda.uhf_sf']
            else:
                sys.modules['pyscf.sftda.uhf_sf'] = saved



if __name__ == '__main__':
    unittest.main()
