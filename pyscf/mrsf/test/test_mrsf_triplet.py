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
'''Triplet MRSF / EMRSF (td.singlet = False).'''

import unittest
import numpy
from pyscf import gto, scf, dft
from pyscf import mrsf
from pyscf.mrsf import rohf_mrsf, mrsf_hop

GEOM = 'O 0.05 0.02 -0.04; H -0.60 0.50 -0.62; H 0.51 -0.55 -0.58'
XC = '0.5*HF+0.5*B88,LYP'
TOL = 1e-10


def setUpModule():
    global mol, mf_hf, mf_gga, mf_lda
    mol = gto.M(atom=GEOM, basis='631g', spin=2, verbose=0, output='/dev/null')
    mf_hf = scf.ROHF(mol).run(conv_tol=1e-12)
    mf_gga = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).run()
    mf_lda = dft.ROKS(mol).set(xc='0.5*HF+0.5*SLATER,VWN', conv_tol=1e-12).run()


def tearDownModule():
    global mol, mf_hf, mf_gga, mf_lda
    mol.stdout.close()
    del mol, mf_hf, mf_gga, mf_lda


def triplet(factory, mf):
    td = factory(mf)
    td.singlet = False
    return td


class KnownValues(unittest.TestCase):
    def test_dim(self):
        cidx, oidx, vidx = rohf_mrsf.orb_indices(mf_gga)
        nc, nv = len(cidx), len(vidx)
        td = triplet(mrsf.TDA_MRSF, mf_gga)
        self.assertEqual(td.dim, (nc + 2) * (nv + 2) - 3)
        self.assertEqual(td.get_ab_mrsf().shape[0], td.dim)
        td = triplet(mrsf.TDA_EMRSF, mf_gga)
        self.assertEqual(td.dim, (nc + 2) * (nv + 2) - 3 + nc * nv)
        self.assertEqual(td.get_ab_mrsf().shape[0], td.dim)

    def test_lr_is_the_reference_triplet(self):
        # L+R is the Ms=0 component of the ROHF triplet: zero energy relative to T at HF
        a = triplet(mrsf.TDA_MRSF, mf_hf).get_ab_mrsf()
        self.assertAlmostEqual(a[0, 0], 0, delta=1e-10)

    def test_matrix_free_equals_dense(self):
        for mf in (mf_hf, mf_gga, mf_lda):
            hyb = rohf_mrsf.hybrid_coeff(mf)
            for expansion in (False, True):
                for spc in (True, False):
                    a = rohf_mrsf.build_matrix(mf, hyb, expansion, spc, singlet=False)
                    vind, hdiag = mrsf_hop.gen_tda_operation(mf, hyb, expansion, spc, singlet=False)
                    b = numpy.asarray(vind(numpy.eye(a.shape[0])))
                    self.assertEqual(hdiag.size, a.shape[0])
                    self.assertAlmostEqual(abs(b - a).max(), 0, delta=TOL, msg=(expansion, spc))

    def test_davidson_equals_eigh(self):
        for factory in (mrsf.TDA_MRSF, mrsf.TDA_EMRSF):
            td = triplet(factory, mf_gga)
            td.nstates = 4
            w = numpy.linalg.eigvalsh(td.get_ab_mrsf())[:4]
            td2 = triplet(factory, mf_gga)
            td2.nstates = 4
            td2.dense_threshold = 0
            e = td2.kernel()[0]
            self.assertTrue(all(td2.converged))
            self.assertAlmostEqual(abs(e - w).max(), 0, delta=1e-8)

    def test_singlet_flag_changes_matrix_and_cache(self):
        td = mrsf.TDA_EMRSF(mf_gga)
        n_singlet = td.get_ab_mrsf().shape[0]
        td.singlet = False
        self.assertEqual(n_singlet - td.get_ab_mrsf().shape[0], 2)

    def test_triplet_transition_properties(self):
        for factory in (mrsf.TDA_MRSF, mrsf.TDA_EMRSF):
            td = triplet(factory, mf_gga)
            td.nstates = 3
            td.kernel()
            t01 = td.trans_rdm1(0, 1)
            self.assertAlmostEqual(abs(t01 - td.trans_rdm1(1, 0).T).max(), 0, delta=1e-12)
            self.assertAlmostEqual(abs(numpy.trace(t01)), 0, delta=1e-10)
            self.assertEqual(td.oscillator_strength().shape, (2,))


if __name__ == '__main__':
    unittest.main()
