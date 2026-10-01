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
        ref = [-0.2779365614, 0.0458543463, 0.1089532602]
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

    def test_davidson_keeps_negative_root(self):
        td = mrsf.TDA_EMRSF(self.mf_dz)
        td.dense_threshold = 0
        td.nstates = 2
        e = td.kernel()[0]
        self.assertLess(e[0], 0)

    def test_lif_reference_energies(self):
        td = mrsf.TDA_EMRSF(self.mf_lif)
        td.nstates = 4
        e = td.kernel()[0]
        ref = [-0.0105054347, -0.0020748345, -0.0020748345, 0.0170434849]
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
        for name in ['transition_quadrupole',
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


class KnownValues_Objects(unittest.TestCase):
    def test_log_name_follows_expansion(self):
        import io
        td = rohf_mrsf.TDA_EMRSF(mf_gga, expansion=False)
        td.nstates = 2
        td.stdout = io.StringIO()
        td.verbose = 4
        td.kernel()
        out = td.stdout.getvalue()
        self.assertIn('MRSF energies', out)
        self.assertNotIn('EMRSF energies', out)

    def test_symmetry_adapted_roks(self):
        mol_s = gto.M(atom=GEOM, basis='631g', spin=2, symmetry=True, verbose=0)
        mf_s = dft.ROKS(mol_s).set(xc=XC, conv_tol=1e-12).run()
        td = mf_s.TDA_EMRSF()
        td.nstates = 3
        ref = mrsf.TDA_EMRSF(mf_gga)
        ref.nstates = 3
        self.assertAlmostEqual(abs(td.kernel()[0] - ref.kernel()[0]).max(), 0, delta=1e-7)

    def test_soscf_reference(self):
        mf_n = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).newton().run()
        td = mrsf.TDA_EMRSF(mf_n)
        td.nstates = 3
        ref = mrsf.TDA_EMRSF(mf_gga)
        ref.nstates = 3
        self.assertAlmostEqual(abs(td.kernel()[0] - ref.kernel()[0]).max(), 0, delta=1e-7)

    def test_as_scanner(self):
        mol2 = gto.M(atom='O 0.05 0.02 -0.04; H -0.70 0.55 -0.70; H 0.58 -0.62 -0.66',
                     basis='631g', spin=2, verbose=0)
        td = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).TDA_EMRSF()
        td.nstates = 2
        scanner = td.as_scanner()
        e_scan = scanner(mol2)
        fresh = dft.ROKS(mol2).set(xc=XC, conv_tol=1e-12).run().TDA_EMRSF()
        fresh.nstates = 2
        fresh.kernel()
        self.assertAlmostEqual(abs(numpy.asarray(e_scan) - fresh.e_tot).max(), 0, delta=1e-7)



class KnownValues_TransitionProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # reference: OpenQP MRSF-TDDFT, examples/other/h2o_rohf_mrsf-s_6-31g_bhhlyp.inp
        mol_q = gto.M(atom='''
            O   0.000000000   0.000000000  -0.041061554
            H  -0.533194329   0.533194329  -0.614469223
            H   0.533194329  -0.533194329  -0.614469223''', basis='631g', spin=2, verbose=0)
        cls.mf_q = dft.ROKS(mol_q).set(xc=XC, conv_tol=1e-12).run()
        cls.mf_q_hf = scf.ROHF(mol_q).run(conv_tol=1e-12)

    def test_matches_openqp(self):
        td = mrsf.TDA_MRSF(self.mf_q)
        td.nstates = 5
        td.kernel()
        exc = (td.e[1:] - td.e[0]) * 27.211386245988
        self.assertAlmostEqual(abs(exc - [8.762908, 10.414620, 10.712630, 14.911981]).max(), 0,
                               delta=1e-3)
        dip = td.transition_dipole()
        self.assertEqual(dip.shape, (4, 3))
        self.assertAlmostEqual(abs(numpy.linalg.norm(dip, axis=1) - [0.2566, 0., 0.6860, 0.9049]).max(),
                               0, delta=2e-4)
        f = td.oscillator_strength()
        self.assertAlmostEqual(abs(f - [0.0141, 0., 0.1235, 0.2991]).max(), 0, delta=2e-4)

    def test_matches_openqp_hf(self):
        # no DFT grid: agreement is limited only by the printed digits of OpenQP
        # (same input with functional= and conv=1e-10)
        td = mrsf.TDA_MRSF(self.mf_q_hf)
        td.nstates = 5
        td.kernel()
        self.assertAlmostEqual(self.mf_q_hf.e_tot, -75.7192307219, delta=1e-7)
        e_tot_ref = [-75.9387930719, -75.6929444561, -75.6109767748, -75.5898191999, -75.4250388416]
        self.assertAlmostEqual(abs(td.e_tot - e_tot_ref).max(), 0, delta=1e-7)
        exc = (td.e[1:] - td.e[0]) * 27.211386245988
        self.assertAlmostEqual(abs(exc - [6.689882, 8.920336, 9.496063, 13.979965]).max(), 0, delta=5e-6)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(numpy.linalg.norm(dip, axis=1) - [0.2620, 0., 0.7319, 0.8605]).max(),
                               0, delta=1e-4)
        self.assertAlmostEqual(abs(td.oscillator_strength() - [0.0113, 0., 0.1246, 0.2536]).max(), 0,
                               delta=1e-4)

    def test_trans_rdm1_properties(self):
        td = mrsf.TDA_MRSF(self.mf_q)
        td.nstates = 4
        td.kernel()
        t01 = td.trans_rdm1(0, 1)
        self.assertAlmostEqual(abs(t01 - td.trans_rdm1(1, 0).T).max(), 0, delta=1e-12)
        self.assertAlmostEqual(abs(numpy.trace(t01)), 0, delta=1e-10)

    def test_emrsf_transition_properties(self):
        td = mrsf.TDA_EMRSF(self.mf_q)
        td.nstates = 5
        td.kernel()
        t01 = td.trans_rdm1(0, 1)
        self.assertAlmostEqual(abs(t01 - td.trans_rdm1(1, 0).T).max(), 0, delta=1e-12)
        self.assertAlmostEqual(abs(numpy.trace(t01)), 0, delta=1e-10)
        self.assertEqual(td.transition_dipole().shape, (4, 3))
        self.assertEqual(td.oscillator_strength().shape, (4,))

    def test_triplet_matches_openqp(self):
        # OpenQP MRSF, h2o_rohf_mrsf-s_6-31g_bhhlyp.inp with [tdhf] multiplicity=3
        td = mrsf.TDA_MRSF(self.mf_q)
        td.singlet = False
        td.nstates = 5
        td.kernel()
        exc = (td.e[1:] - td.e[0]) * 27.211386245988
        self.assertAlmostEqual(abs(exc - [1.668297, 1.806295, 5.764427, 13.033019]).max(), 0, delta=1e-3)
        self.assertAlmostEqual((td.e[0]) * 27.211386245988, 0.831735, delta=1e-3)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(numpy.linalg.norm(dip, axis=1) - [0.1741, 1.8050, 0., 0.4204]).max(),
                               0, delta=2e-4)
        self.assertAlmostEqual(abs(td.oscillator_strength() - [0.0012, 0.1442, 0., 0.0564]).max(), 0,
                               delta=2e-4)

    def test_triplet_matches_openqp_hf(self):
        # OpenQP MRSF, same input with [tdhf] multiplicity=3 and functional=
        td = mrsf.TDA_MRSF(self.mf_q_hf)
        td.singlet = False
        td.nstates = 5
        td.kernel()
        e_tot_ref = [-75.7215916830, -75.6370429621, -75.6284048562, -75.4812117950, -75.1710815246]
        self.assertAlmostEqual(abs(td.e_tot - e_tot_ref).max(), 0, delta=1e-7)
        exc = (td.e[1:] - td.e[0]) * 27.211386245988
        self.assertAlmostEqual(abs(exc - [2.300688, 2.535743, 6.541070, 14.980145]).max(), 0, delta=5e-6)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(numpy.linalg.norm(dip, axis=1) - [0.1624, 1.7988, 0., 0.3924]).max(),
                               0, delta=1e-4)
        self.assertAlmostEqual(abs(td.oscillator_strength() - [0.0015, 0.2010, 0., 0.0565]).max(), 0,
                               delta=1e-4)



if __name__ == '__main__':
    unittest.main()
