#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

'''Tests of the GBCI solvers: block sigma, generalized Davidson, EDIIS + DIIS
baths and pair one-electron Hamiltonians.'''

import unittest

import numpy as np
from pyscf import gto, lib, scf
from pyscf.fci import cistring, spin_op
from pyscf import gbci
from pyscf.gbci import direct_gbci
from pyscf.gbci.gbci import h1e_for_gbci


def setUpModule():
    global mol, mf, mol_t, mf_t
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='6-31g', verbose=0, output='/dev/null')
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mol_t = gto.M(atom='C 0 0 0; H 0 0.9 0.6; H 0 -0.9 0.6', basis='6-31g',
                  spin=2, verbose=0, output='/dev/null')
    mf_t = scf.ROHF(mol_t).run(conv_tol=1e-12)


def tearDownModule():
    global mol, mf, mol_t, mf_t
    mol.stdout.close()
    mol_t.stdout.close()
    del mol, mf, mol_t, mf_t


def _gbci(mf, ncas, nelecas, group_a=None, **solver):
    mc = gbci.gbci(mf, ncas, nelecas, group_a=group_a)
    for key, val in solver.items():
        setattr(mc.fcisolver, key, val)
    return mc


def _hamiltonian(mc):
    inter = mc.get_gbci_intermediates()
    h1e, ecore = mc.get_h1cas(mc.get_active_dm(mc.mo_coeff),
                              inter['mo_list'], inter['dmet_core_list'])
    eri = mc.get_h2eff(mc.mo_coeff)
    return h1e, eri, inter['conf_info_list'], inter['ov_list'], ecore


def _hdiag_loop(h1e, eri, ncas, nelecas, conf_info_list, ecore_list):
    '''make_hdiag in its loop form.'''
    from pyscf import ao2mo
    occslista = cistring.gen_occslst(range(ncas), nelecas[0])
    occslistb = cistring.gen_occslst(range(ncas), nelecas[1])
    eri = ao2mo.restore(1, eri, ncas)
    diagj = np.einsum('iijj->ij', eri)
    diagk = np.einsum('ijji->ij', eri)
    hdiag = []
    for str0a, aocc in enumerate(occslista):
        for str0b, bocc in enumerate(occslistb):
            p = conf_info_list[str0a, str0b]
            e1 = h1e[p, p, aocc, aocc].sum() + h1e[p, p, bocc, bocc].sum()
            e2 = (diagj[aocc][:, aocc].sum() + diagj[aocc][:, bocc].sum()
                  + diagj[bocc][:, aocc].sum() + diagj[bocc][:, bocc].sum()
                  - diagk[aocc][:, aocc].sum() - diagk[bocc][:, bocc].sum())
            hdiag.append(e1 + e2 * .5 + ecore_list[p])
    return np.array(hdiag)


def _h1e_loop(mc, mo_list, dmet_core_list):
    '''Pair one-electron Hamiltonians, one get_veff per pair.'''
    ncas, ncore = mc.ncas, mc.ncore
    p = dmet_core_list.shape[0]
    mo_cas = mo_list[0][:, ncore:ncore + ncas]
    hcore = mc.get_hcore()
    ha1e = mo_cas.T @ hcore @ mo_cas
    h1e = np.zeros((p, p, ncas, ncas))
    ecore = np.zeros(p)
    for i in range(p):
        for j in range(i + 1):
            v = mc.get_veff(dm=2 * dmet_core_list[i, j], hermi=0)
            h1e[i, j] = ha1e + mo_cas.T @ v.T @ mo_cas
            h1e[j, i] = h1e[i, j].T
            if i == j:
                ecore[i] = (np.einsum('ab,ab', dmet_core_list[i, i], v)
                            + mc.energy_nuc()
                            + 2 * np.einsum('ab,ab', dmet_core_list[i, i], hcore))
    return h1e, ecore


class KnownValues(unittest.TestCase):
    def test_sigma_and_hdiag(self):
        mc = _gbci(mf, 4, (2, 2))
        h1e, eri, conf, ov, ecore = _hamiltonian(mc)
        self.assertGreater(len(np.unique(conf)), 5)
        self.assertTrue((abs(ov - np.eye(len(ov))) > 1e-3).any())
        hdiag = direct_gbci.make_hdiag(h1e, eri, 4, (2, 2), conf, ecore)
        np.testing.assert_allclose(
            hdiag, _hdiag_loop(h1e, eri, 4, (2, 2), conf, ecore), atol=1e-12)
        erieff = direct_gbci.absorb_h1e(h1e, eri, 4, (2, 2), .5)
        c = np.random.default_rng(3).standard_normal(conf.shape)
        ref = direct_gbci.contract_h(erieff, c, 4, (2, 2), conf, ov, ecore)
        new = direct_gbci.contract_h_block(h1e, eri, c, 4, (2, 2), conf, ov,
                                           ecore)
        np.testing.assert_allclose(new, ref.reshape(c.shape), atol=1e-10)

    def test_pspace_and_spin_blocks(self):
        mc = _gbci(mf, 4, (2, 2), group_a={'mo': [[0, 1], [2, 3]]})
        h1e, eri, conf, ov, ecore = _hamiltonian(mc)
        ndet = conf.size
        sigma = direct_gbci.BlockSigma(h1e, eri, 4, (2, 2), conf, ov, ecore)
        hfull = np.array([sigma.contract(np.eye(ndet)[k].reshape(conf.shape)).ravel()
                          for k in range(ndet)]).T
        hdiag = direct_gbci.make_hdiag(h1e, eri, 4, (2, 2), conf, ecore)
        addr = direct_gbci.spin_complete_pspace(hdiag, 4, (2, 2), 12)
        h0 = direct_gbci.pspace_hamiltonian(h1e, eri, 4, (2, 2), conf, ov,
                                            ecore, addr, hdiag)
        np.testing.assert_allclose(h0, hfull[np.ix_(addr, addr)], atol=1e-10)
        s2full = np.array([spin_op.contract_ss(np.eye(ndet)[k].reshape(conf.shape),
                                               4, (2, 2)).ravel()
                           for k in range(ndet)]).T
        s2, complete = direct_gbci.s2_block(4, (2, 2), addr)
        self.assertTrue(complete)
        np.testing.assert_allclose(s2, s2full[np.ix_(addr, addr)], atol=1e-12)

    def test_solver_paths(self):
        # every solver path gives the same roots
        ref = None
        for opts in ({'sigma_kernel': 'contract_h', 'pspace_size': 0,
                      'guess_noise': 0},
                     {},
                     {'sigma_kernel': 'block'},
                     {'davidson_only': True},
                     {'pspace_size': 0}):
            mc = _gbci(mf, 4, (2, 2), group_a={'mo': [[0, 1], [2, 3]]},
                       nroots=3, conv_tol=1e-10, **opts)
            mc.fix_spin_(ss=0, shift=.5)
            e = np.asarray(mc.kernel()[0])
            if ref is None:
                ref = e
            np.testing.assert_allclose(e, ref, atol=1e-8)

    def test_triplet_penalty(self):
        mc = _gbci(mf_t, 6, (4, 2), nroots=2, conv_tol=1e-10,
                   sigma_kernel='contract_h', pspace_size=0, guess_noise=0)
        mc.fix_spin_(ss=2, shift=.5)
        e_ref = np.asarray(mc.kernel()[0])
        mc = _gbci(mf_t, 6, (4, 2), nroots=2, conv_tol=1e-10, pspace_size=30)
        mc.fix_spin_(ss=2, shift=.5)
        e, _, ci = mc.kernel()
        np.testing.assert_allclose(e, e_ref, atol=1e-8)
        for c in ci:
            self.assertAlmostEqual(spin_op.spin_square0(c, 6, (4, 2))[0], 2, 6)

    def test_ediis_bath(self):
        # EDIIS + DIIS baths = damped iterations + Newton baths
        for group_a in (None, {'mo': [[0, 1], [2, 3]]}):
            res = []
            for ediis in (False, True):
                mc = _gbci(mf, 4, (2, 2), group_a=group_a)
                mc._fasscf.mixed_routine_options = {'ediis': ediis}
                mc._fasscf.conv_tol = 1e-11
                mo_list, mo_e, po_list, group = mc.optimize_mo()
                res.append((mo_list, mo_e))
            ncore = mc.ncore
            s1e = mf.get_ovlp()
            for c0, c1 in zip(res[0][0], res[1][0]):
                p0 = c0[:, :ncore] @ c0[:, :ncore].T
                p1 = c1[:, :ncore] @ c1[:, :ncore].T
                np.testing.assert_allclose(s1e @ p0 @ s1e, s1e @ p1 @ s1e,
                                           atol=1e-6)
            inact = np.r_[0:ncore, ncore + 4:mf.mo_coeff.shape[1]]
            np.testing.assert_allclose(res[0][1][:, inact], res[1][1][:, inact],
                                       atol=1e-6)

    def test_h1e_fast_path(self):
        for mf1 in (mf, scf.RHF(mol).density_fit(auxbasis='weigend').run(conv_tol=1e-12)):
            mc = _gbci(mf1, 4, (2, 2))
            inter = mc.get_gbci_intermediates()
            h1e, ecore = h1e_for_gbci(mc, mc.get_active_dm(mc.mo_coeff),
                                      inter['mo_list'], inter['dmet_core_list'])
            h1e_ref, ecore_ref = _h1e_loop(mc, inter['mo_list'],
                                           inter['dmet_core_list'])
            np.testing.assert_allclose(h1e, h1e_ref, atol=1e-10)
            np.testing.assert_allclose(ecore, ecore_ref, atol=1e-10)


if __name__ == '__main__':
    print('Tests for the GBCI solvers')
    unittest.main()
