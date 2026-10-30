#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import unittest

import numpy as np
from pyscf import gto, mcpdft, scf
from pyscf.gbci import gbpdft, gbsrpdft, lgbpdft

# LiF at 2.5 A, CAS(2,4)/6-31G (F 2pz and three Li-centred virtuals), Li / F
# atom groups, four singlet roots (roots 3 and 4 are a degenerate pair, so
# the model space must hold both).
_MC = None


def setUpModule():
    global _MC, _MF, _MO, _INTER, _E, _CI
    mol = gto.M(atom="Li 0 0 0; F 0 0 2.5", basis="6-31g", verbose=0)
    _MF = scf.RHF(mol).run(conv_tol=1e-11)
    mc = gbpdft.GBCI(_MF, "tPBE", 4, (1, 1), group_a={"atom": [[0], [1]]},
                     grids_level=3)
    mc.verbose = 0
    mc.fcisolver.spin = 0
    mc.fix_spin_(ss=0.0, shift=1.0)
    # the HOMO (F 2pz) and the three lowest Li-centred virtuals
    nocc = mol.nelectron // 2
    lab = mol.ao_labels()
    s = mol.intor_symmetric("int1e_ovlp")
    li = [i for i, x in enumerate(lab) if x.startswith("0 Li")]
    cs = s @ _MF.mo_coeff
    wli = np.einsum("ai,ai->i", _MF.mo_coeff[li], cs[li])
    vir = [i for i in range(nocc, mol.nao) if wli[i] > 0.5][:3]
    cas = [nocc - 1] + vir
    from pyscf.mcscf import addons
    _MO = addons.sort_mo(mc, _MF.mo_coeff, [i + 1 for i in cas], 1)
    mc.mo_coeff = _MO
    _INTER = mc.get_gbci_intermediates(_MO)
    _E, _CI = gbsrpdft.solve(mc, gbsrpdft.full_hamiltonian(mc, _MO, _INTER),
                             _INTER, 4)
    _MC = mc


def tearDownModule():
    global _MC, _MF
    _MC = _MF = None


class KnownValues(unittest.TestCase):

    def test_long_range_limit(self):
        # omega -> large: H^lr -> H and E^sr_Hxc -> 0 as 1/omega^2, so
        # GB-srPDFT -> GBCI (libxc's erf functionals fail beyond ~200)
        d = [abs(gbsrpdft.kernel_ss(_MC, _CI, mo_coeff=_MO, omega=w,
                                    intermediates=_INTER)["e_states"]
                 - _E).max() for w in (50.0, 200.0)]
        self.assertLess(d[1], 5e-4)
        self.assertTrue(10 < d[0] / d[1] < 20)

    def test_multistate_one_state(self):
        # a one-state model space: XMS and FCD reduce to the SS energy
        ss = gbsrpdft.kernel_ss(_MC, _CI[:1], mo_coeff=_MO,
                                intermediates=_INTER)["e_states"]
        xms = gbsrpdft.kernel_xms_gbci(_MC, _CI[:1], mo_coeff=_MO,
                                       intermediates=_INTER)["e_states"]
        w = gbsrpdft.fragment_weights(_MC.mol, _MO, _MC.ncore, _MC.ncas, [0])
        fcd = gbsrpdft.kernel_fcd_gbci(_MC, _CI[:1], w, mo_coeff=_MO,
                                       intermediates=_INTER)["e_states"]
        self.assertAlmostEqual(xms[0], ss[0], 9)
        self.assertAlmostEqual(fcd[0], ss[0], 9)

    def test_multistate_trace(self):
        # H^eff is symmetric and its eigenvalues sum to its trace
        w = gbsrpdft.fragment_weights(_MC.mol, _MO, _MC.ncore, _MC.ncas, [0])
        for r in (gbsrpdft.kernel_xms_gbci(_MC, _CI, mo_coeff=_MO,
                                           intermediates=_INTER),
                  gbsrpdft.kernel_fcd_gbci(_MC, _CI, w, mo_coeff=_MO,
                                           intermediates=_INTER),
                  gbsrpdft.kernel_fcd_xms(_MC, _CI, w, mo_coeff=_MO,
                                          intermediates=_INTER, otxc="tPBE0")):
            h = r["heff"]
            self.assertAlmostEqual(abs(h - h.T).max(), 0, 12)
            self.assertAlmostEqual(np.trace(h), np.sum(r["e_states"]), 9)

    def test_lpdft_plin_diagonal(self):
        # PLIN diagonal is the state-specific GB-PDFT energy, exact and
        # approximate Pi; LIN of one state likewise
        ctx = gbsrpdft._Context(_MC, _MO, _INTER, gbsrpdft.DEFAULT_OMEGA)
        for fx in ("tPBE", "tPBE0"):
            ot = mcpdft.otfnal.get_transfnal(_MC.mol, fx)
            ot.grids = _MC.otfnal.grids
            for ontop in ("exact", "cas_cumulant"):
                ss = np.array([ctx.e_pdft(c, ot=ot, ontop=ontop) for c in _CI])
                pl = lgbpdft.kernel(_MC, _CI, mo_coeff=_MO, ot=ot,
                                    intermediates=_INTER, method="PLIN",
                                    ontop=ontop)
                self.assertAlmostEqual(abs(np.diag(pl["heff"]) - ss).max(),
                                       0, 9)
                one = lgbpdft.kernel(_MC, _CI[:1], mo_coeff=_MO, ot=ot,
                                     intermediates=_INTER, method="LIN",
                                     ontop=ontop)
                self.assertAlmostEqual(one["e_states"][0], ss[0], 9)

    def test_lpdft_linear_term(self):
        # the linear on-top term against a finite difference of E_ot along
        # the reference -> state 1 direction
        ot = mcpdft.otfnal.get_transfnal(_MC.mol, "tPBE")
        ot.grids = _MC.otfnal.grids
        data = _MC.make_otpd_intermediates(_MO, intermediates=_INTER)
        for ontop in ("exact", "cas_cumulant"):
            grid = lgbpdft._Grid(_MC, ot, _MO, data, _CI, _MC.max_memory,
                                 ontop)
            w0 = np.ones(4) / 4
            e0, lin, lref = lgbpdft._ot_terms(ot, grid, w0)
            eps = 1e-4
            e1 = lgbpdft._ot_terms(ot, grid,
                                   (1 - eps) * w0 + eps * np.eye(4)[1])[0]
            d = lin[1] - lref
            self.assertAlmostEqual((e1 - e0) / eps, d,
                                   delta=1e-3 * max(1.0, abs(d)))

    def test_ontop_approx_close(self):
        # approximate Pi stays close to the exact one in the sr energy
        a = gbsrpdft.kernel_ss(_MC, _CI, mo_coeff=_MO, intermediates=_INTER)
        b = gbsrpdft.kernel_ss(_MC, _CI, mo_coeff=_MO, intermediates=_INTER,
                               ontop="cas_cumulant")
        self.assertLess(abs(a["e_states"] - b["e_states"]).max(), 2e-3)


if __name__ == "__main__":
    print("Full Tests for GB-srPDFT and L-GBPDFT")
    unittest.main()
