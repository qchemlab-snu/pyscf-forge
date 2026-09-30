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
'''Tests of the pyscf.mrsf package interface (forge layout).'''

import unittest
import numpy
from pyscf import gto, scf, dft
from pyscf import mrsf
from pyscf.mrsf import rohf_mrsf

GEOM = 'O 0.05 0.02 -0.04; H -0.60 0.50 -0.62; H 0.51 -0.55 -0.58'
XC = '0.5*HF+0.5*B88,LYP'


def setUpModule():
    global mol, mf_hf, mf_gga, mf_df
    mol = gto.M(atom=GEOM, basis='631g', spin=2, verbose=0, output='/dev/null')
    mf_hf = scf.ROHF(mol).run(conv_tol=1e-12)
    mf_gga = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).run()
    mf_df = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).density_fit(auxbasis='cc-pvdz-jkfit').run()


def tearDownModule():
    global mol, mf_hf, mf_gga, mf_df
    mol.stdout.close()
    del mol, mf_hf, mf_gga, mf_df


class KnownValues_Interface(unittest.TestCase):
    def test_factories(self):
        td = mrsf.TDA_MRSF(mf_gga)
        self.assertIsInstance(td, rohf_mrsf.TDA_MRSF)
        self.assertFalse(td.expansion)
        td = mrsf.TDA_EMRSF(mf_gga)
        self.assertIsInstance(td, rohf_mrsf.TDA_EMRSF)
        self.assertTrue(td.expansion)
        self.assertIs(rohf_mrsf.EMRSF_TDA, rohf_mrsf.TDA_EMRSF)

    def test_registration_roks_and_df(self):
        for mf in (mf_hf, mf_gga, mf_df):
            self.assertIsInstance(mf.TDA_EMRSF(), rohf_mrsf.TDA_EMRSF)
            self.assertIsInstance(mf.TDA_MRSF(), rohf_mrsf.TDA_MRSF)

    def test_is_tdbase(self):
        from pyscf.tdscf.rhf import TDBase
        self.assertIsInstance(mrsf.TDA_EMRSF(mf_gga), TDBase)

    def test_expansion_keyword_still_works(self):
        self.assertFalse(rohf_mrsf.TDA_EMRSF(mf_gga, expansion=False).expansion)

    def test_reference_energies(self):
        td = mrsf.TDA_MRSF(mf_gga)
        td.nstates = 3
        e = td.kernel()[0]
        ref = [-0.2716684878, 0.0458581413, 0.1089593088]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)
        td = mrsf.TDA_EMRSF(mf_gga)
        td.nstates = 3
        e = td.kernel()[0]
        ref = [-0.2780041182, 0.0458543900, 0.1085821003]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)
        self.assertAlmostEqual(abs(td.e_tot - (mf_gga.e_tot + e)).max(), 0, delta=1e-12)


class KnownValues_Kernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol_dz = gto.M(atom=GEOM, basis='ccpvdz', spin=2, verbose=0, output='/dev/null')
        cls.mf_dz = dft.ROKS(mol_dz).set(xc=XC, conv_tol=1e-12).run()
        mol_lif = gto.M(atom='Li 0 0 0; F 0 0 4.4', basis='631g', spin=2, verbose=0,
                        output='/dev/null')
        cls.mf_lif = dft.ROKS(mol_lif).set(xc='0.76*HF+0.24*B88,LYP', init_guess='huckel',
                                           conv_tol=1e-12).run()

    def test_kernel_paths_agree(self):
        for mf in (self.mf_dz, self.mf_lif):
            dense = mrsf.TDA_EMRSF(mf)
            dense.nstates = 5
            dense.dense_threshold = 10**9
            e_dense = dense.kernel()[0]
            td = mrsf.TDA_EMRSF(mf)
            td.nstates = 5
            td.dense_threshold = 0
            e = td.kernel()[0]
            self.assertTrue(all(td.converged))
            self.assertAlmostEqual(abs(e - e_dense).max(), 0, delta=1e-8)

    def test_lr_eigh_keeps_negative_root(self):
        td = mrsf.TDA_EMRSF(self.mf_dz)
        td.dense_threshold = 0
        td.nstates = 2
        e = td.kernel()[0]
        self.assertLess(e[0], 0)

    def test_lif_reference_energies(self):
        td = mrsf.TDA_EMRSF(self.mf_lif)
        td.nstates = 4
        e = td.kernel()[0]
        ref = [-0.0105491698, -0.0084618841, -0.0084618841, 0.0168605453]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)

    def test_chkfile_roundtrip(self):
        import tempfile
        from pyscf import lib
        with tempfile.NamedTemporaryFile(suffix='.chk') as f:
            td = mrsf.TDA_EMRSF(mf_gga)
            td.chkfile = f.name
            td.nstates = 3
            e, xy = td.kernel()
            self.assertAlmostEqual(abs(lib.chkfile.load(f.name, 'tddft/e') - e).max(), 0,
                                   delta=1e-14)
            xy_saved = lib.chkfile.load(f.name, 'tddft/xy')
            self.assertAlmostEqual(abs(numpy.asarray(xy_saved[0][0]) - xy[0][0]).max(), 0,
                                   delta=1e-14)

    def test_analyze(self):
        import io
        for factory in (mrsf.TDA_MRSF, mrsf.TDA_EMRSF):
            td = factory(mf_gga)
            td.nstates = 3
            td.kernel()
            td.stdout = io.StringIO()
            td.verbose = 4
            td.analyze()
            out = td.stdout.getvalue()
            self.assertIn('State 0', out)
            self.assertIn('G   ', out)
            self.assertIn('weights', out)
            self.assertEqual('CV_ext' in out, factory is mrsf.TDA_EMRSF)

    def test_nuc_grad_method_raises(self):
        self.assertRaises(NotImplementedError, mrsf.TDA_EMRSF(mf_gga).nuc_grad_method)



class KnownValues_ReviewFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol_lif = gto.M(atom='Li 0 0 0; F 0 0 4.4', basis='631g', spin=2, verbose=0,
                        output='/dev/null')
        cls.mf_lif = dft.ROKS(mol_lif).set(xc='0.76*HF+0.24*B88,LYP', init_guess='huckel',
                                           conv_tol=1e-12).run()

    def test_matrix_free_matvec_count(self):
        td = mrsf.TDA_EMRSF(self.mf_lif)
        td.nstates = 3
        td.dense_threshold = 0
        gen_vind = td.gen_vind
        count = [0]

        def counting_gen_vind():
            vind, hdiag = gen_vind()

            def vind_counted(xs):
                xs = numpy.asarray(xs)
                count[0] += len(xs.reshape(-1, hdiag.size))
                return vind(xs)
            return vind_counted, hdiag
        td.gen_vind = counting_gen_vind
        td.kernel()
        self.assertTrue(all(td.converged))
        self.assertLess(count[0], 40)

    def test_reset_propagates_to_scf(self):
        mol2 = gto.M(atom='O 0.05 0.02 -0.04; H -0.70 0.55 -0.70; H 0.58 -0.62 -0.66',
                     basis='631g', spin=2, verbose=0)
        mf = dft.ROKS(mol).set(xc=XC)
        td = mrsf.TDA_EMRSF(mf)
        td.reset(mol2)
        self.assertIs(td.mol, mol2)
        self.assertIs(td._scf.mol, mol2)

    def test_frozen_and_wfnsym_rejected(self):
        td = mrsf.TDA_EMRSF(mf_gga)
        td.frozen = [0]
        self.assertRaises(NotImplementedError, td.kernel)
        td = mrsf.TDA_EMRSF(mf_gga)
        td.wfnsym = 'A1'
        self.assertRaises(NotImplementedError, td.kernel)



class KnownValues_Conventions(unittest.TestCase):
    def test_unsupported_properties_raise_clear_error(self):
        td = mrsf.TDA_EMRSF(mf_gga)
        td.nstates = 2
        td.kernel()
        for name in ['oscillator_strength', 'transition_dipole', 'transition_quadrupole',
                     'transition_octupole', 'transition_velocity_dipole',
                     'transition_velocity_quadrupole', 'transition_velocity_octupole',
                     'transition_magnetic_dipole', 'transition_magnetic_quadrupole', 'get_nto']:
            with self.assertRaises(NotImplementedError) as ctx:
                getattr(td, name)()
            self.assertIn('MRSF', str(ctx.exception), msg=name)

    def test_get_ab_mrsf(self):
        hfx = 0.5
        a_ref = rohf_mrsf.build_matrix(mf_gga, hfx, expansion=True)
        self.assertAlmostEqual(abs(mrsf.get_ab_mrsf(mf_gga) - a_ref).max(), 0, delta=1e-12)
        self.assertAlmostEqual(abs(rohf_mrsf.get_ab_mrsf(mf_gga, expansion=False)
                                   - rohf_mrsf.build_matrix(mf_gga, hfx, expansion=False)).max(),
                               0, delta=1e-12)
        td = mrsf.TDA_EMRSF(mf_gga)
        self.assertAlmostEqual(abs(td.get_ab_mrsf() - a_ref).max(), 0, delta=1e-12)
        # names already pushed keep working
        self.assertAlmostEqual(abs(td.get_ab() - a_ref).max(), 0, delta=1e-12)
        self.assertAlmostEqual(abs(rohf_mrsf.get_ab(mf_gga, hfx) - a_ref).max(), 0, delta=1e-12)


    def test_hyb_naming_follows_pyscf(self):
        td = mrsf.TDA_EMRSF(mf_gga)
        self.assertIsNone(td.hyb)
        self.assertAlmostEqual(td.get_hyb(), 0.5, delta=1e-14)
        td.hyb = 0.3
        self.assertAlmostEqual(td.get_hyb(), 0.3, delta=1e-14)
        # pushed names keep working
        self.assertAlmostEqual(td.hfx, 0.3, delta=1e-14)
        td.hfx = 0.4
        self.assertAlmostEqual(td.hyb, 0.4, delta=1e-14)
        self.assertAlmostEqual(td.get_hfx(), 0.4, delta=1e-14)
        a = rohf_mrsf.get_ab_mrsf(mf_gga, hyb=0.5, expansion=False)
        self.assertAlmostEqual(abs(a - rohf_mrsf.build_matrix(mf_gga, 0.5, expansion=False)).max(),
                               0, delta=1e-12)

    def test_gen_tda_operation_name(self):
        from pyscf.mrsf import mrsf_hop
        self.assertIs(mrsf_hop.gen_hop, mrsf_hop.gen_tda_operation)
        vind, hdiag = mrsf_hop.gen_tda_operation(mf_gga, hyb=0.5, expansion=False)
        a = rohf_mrsf.build_matrix(mf_gga, 0.5, expansion=False)
        self.assertAlmostEqual(abs(vind(numpy.eye(hdiag.size)) - a).max(), 0, delta=1e-10)


if __name__ == '__main__':
    unittest.main()
