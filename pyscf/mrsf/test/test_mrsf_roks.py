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

import unittest
import numpy as np
from pyscf import gto, scf, dft
from pyscf import mrsf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()
        cls.mf_hf = scf.ROHF(cls.mol).run(conv_tol=1e-12)
        cls.mf_bhhlyp = dft.ROKS(cls.mol).set(xc='0.5*HF+0.5*B88,LYP', conv_tol=1e-12).run()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf_hf, cls.mf_bhhlyp

    def kernel(self, td, nstates=4):
        td.nstates = nstates
        return td.kernel()[0]

    def test_mrsf_hf(self):
        e = self.kernel(mrsf.TDA_MRSF(self.mf_hf))
        ref = [-0.2149853002, 0.0264500016, 0.1074499713, 0.1332145917]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)

    def test_emrsf_hf(self):
        e = self.kernel(mrsf.TDA_EMRSF(self.mf_hf))
        ref = [-0.2383341019, 0.0264367386, 0.0966347311, 0.1074310566]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)

    def test_mrsf_bhhlyp(self):
        e = self.kernel(mrsf.TDA_MRSF(self.mf_bhhlyp))
        ref = [-0.2714772549, 0.0456474362, 0.1052727777, 0.1200784805]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)

    def test_emrsf_bhhlyp(self):
        e = self.kernel(mrsf.TDA_EMRSF(self.mf_bhhlyp))
        ref = [-0.2780226768, 0.0456436288, 0.1052674108, 0.1128193133]
        self.assertAlmostEqual(abs(e - ref).max(), 0, delta=1e-6)

    def test_matrix_free_equals_dense(self):
        for mf in (self.mf_hf, self.mf_bhhlyp):
            td = mrsf.TDA_EMRSF(mf)
            td.dense_threshold = 0
            e = self.kernel(td)
            self.assertTrue(all(td.converged))
            w = np.linalg.eigvalsh(td.get_ab_mrsf())[:4]
            self.assertAlmostEqual(abs(e - w).max(), 0, delta=1e-8)

    def test_emrsf_stabilizes_ground_state(self):
        e_mrsf = self.kernel(mrsf.TDA_MRSF(self.mf_bhhlyp), 1)
        e_emrsf = self.kernel(mrsf.TDA_EMRSF(self.mf_bhhlyp), 1)
        self.assertLess(e_emrsf[0], e_mrsf[0])


if __name__ == "__main__":
    print("Full Tests for MRSF/EMRSF TDA on ROHF/ROKS references")
    unittest.main()
