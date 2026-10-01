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
'''MRSF and EMRSF do not depend on how the core and the virtual orbitals are chosen
within their subspaces.'''

import unittest
import numpy
from pyscf import gto, scf, dft
from pyscf import mrsf
from pyscf.mrsf import rohf_mrsf, mrsf_hop

GEOM = 'O 0.05 0.02 -0.04; H -0.60 0.50 -0.62; H 0.51 -0.55 -0.58'
XC = '0.5*HF+0.5*B88,LYP'


def setUpModule():
    global mol, mf_hf, mf_gga
    mol = gto.M(atom=GEOM, basis='631g', spin=2, verbose=0, output='/dev/null')
    mf_hf = scf.ROHF(mol).run(conv_tol=1e-12)
    mf_gga = dft.ROKS(mol).set(xc=XC, conv_tol=1e-12).run()


def tearDownModule():
    global mol, mf_hf, mf_gga
    mol.stdout.close()
    del mol, mf_hf, mf_gga


def rotated(mf, seed=1):
    '''Same reference with the core and the virtual orbitals mixed among themselves.'''
    cidx, oidx, vidx = rohf_mrsf.orb_indices(mf)
    rng = numpy.random.RandomState(seed)
    c = mf.mo_coeff.copy()
    for idx in (cidx, vidx):
        c[:, idx] = c[:, idx].dot(numpy.linalg.qr(rng.randn(len(idx), len(idx)))[0])
    mf2 = mf.copy()
    mf2.mo_coeff = c
    return mf2


class KnownValues(unittest.TestCase):
    def test_dense(self):
        for mf in (mf_hf, mf_gga):
            hyb = rohf_mrsf.hybrid_coeff(mf)
            for singlet in (True, False):
                for expansion in (False, True):
                    e = [numpy.linalg.eigvalsh(rohf_mrsf.build_matrix(m, hyb, expansion, singlet=singlet))[:8]
                         for m in (mf, rotated(mf))]
                    self.assertAlmostEqual(abs(e[0] - e[1]).max(), 0, delta=1e-10, msg=(singlet, expansion))

    def test_kernel(self):
        for singlet in (True, False):
            e = []
            for m in (mf_gga, rotated(mf_gga, 2)):
                td = mrsf.TDA_EMRSF(m)
                td.singlet = singlet
                td.nstates = 5
                td.dense_threshold = 0
                e.append(td.kernel()[0])
            self.assertAlmostEqual(abs(e[0] - e[1]).max(), 0, delta=1e-8, msg=singlet)

    def test_matrix_free_equals_dense(self):
        for mf in (mf_hf, mf_gga):
            hyb = rohf_mrsf.hybrid_coeff(mf)
            for singlet in (True, False):
                a = rohf_mrsf.build_matrix(mf, hyb, singlet=singlet)
                vind, hdiag = mrsf_hop.gen_tda_operation(mf, hyb, singlet=singlet)
                self.assertAlmostEqual(abs(numpy.asarray(vind(numpy.eye(a.shape[0]))) - a).max(), 0,
                                       delta=1e-10, msg=singlet)
                self.assertEqual(hdiag.size, a.shape[0])


if __name__ == '__main__':
    unittest.main()
